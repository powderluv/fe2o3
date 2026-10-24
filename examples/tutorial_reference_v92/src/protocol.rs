use serde::{Deserialize, Serialize};

pub const MAX_REQUEST: usize = 16 * 1024 * 1024;
pub const MAX_RESPONSE: usize = 16 * 1024 * 1024;
const MAX_DATA: usize = 4 * 1024 * 1024;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub schema: String,
    pub kernel: String,
    pub grid: [u64; 3],
    pub workgroup: [u32; 3],
    pub arguments: Vec<Argument>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Argument {
    Buffer {
        element: String,
        access: String,
        alignment: u32,
        bytes: String,
    },
    Scalar {
        #[serde(rename = "type")]
        ty: String,
        bits: String,
    },
}

pub fn encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(2 + 2 * bytes.len());
    output.push_str("0x");
    for byte in bytes {
        output.push(HEX[usize::from(byte >> 4)] as char);
        output.push(HEX[usize::from(byte & 15)] as char);
    }
    output
}

pub fn decode(text: &str) -> Result<Vec<u8>, String> {
    let body = text.strip_prefix("0x").ok_or("missing hex prefix")?;
    if body.len() % 2 != 0
        || body.len() > MAX_DATA * 2
        || !body
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err("noncanonical or oversized hex bytes".to_owned());
    }
    body.as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            let digit = |b: u8| if b <= b'9' { b - b'0' } else { b - b'a' + 10 };
            Ok(digit(pair[0]) * 16 + digit(pair[1]))
        })
        .collect()
}

pub fn width(element: &str) -> Result<usize, String> {
    match element {
        "u8" => Ok(1),
        "u16" | "bf16" => Ok(2),
        "u32" | "i32" | "f32" => Ok(4),
        "u64" | "i64" | "index" => Ok(8),
        _ => Err("unsupported reference scalar type".to_owned()),
    }
}

impl Request {
    pub fn new(kernel: &str, workgroup: u32, groups: u64, arguments: Vec<Argument>) -> Self {
        Self {
            schema: "fe2o3-simulation-request-v1".to_owned(),
            kernel: kernel.to_owned(),
            grid: [groups * u64::from(workgroup), 1, 1],
            workgroup: [workgroup, 1, 1],
            arguments,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema != "fe2o3-simulation-request-v1"
            || self.kernel.is_empty()
            || self.kernel.len() > 256
            || !self
                .kernel
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'_')
            || self.arguments.len() > 64
            || self.grid.contains(&0)
            || self.workgroup.contains(&0)
            || self
                .grid
                .iter()
                .try_fold(1u64, |n, x| n.checked_mul(*x))
                .is_none_or(|n| n > 65536)
            || self
                .workgroup
                .iter()
                .try_fold(1u32, |n, x| n.checked_mul(*x))
                .is_none_or(|n| n > 1024)
        {
            return Err("reference request header bound".to_owned());
        }
        let mut total = 0usize;
        for argument in &self.arguments {
            match argument {
                Argument::Buffer {
                    element,
                    access,
                    alignment,
                    bytes,
                } => {
                    let size = width(element)?;
                    let data = decode(bytes)?;
                    if data.len() % size != 0
                        || *alignment as usize != size
                        || !matches!(access.as_str(), "read_only" | "write_only" | "read_write")
                    {
                        return Err("invalid reference buffer".to_owned());
                    }
                    total = total
                        .checked_add(data.len())
                        .ok_or("reference extent overflow")?;
                }
                Argument::Scalar { ty, bits } => {
                    if decode(bits)?.len() != width(ty)? {
                        return Err("scalar width mismatch".to_owned());
                    }
                }
            }
        }
        if total > MAX_DATA {
            return Err("aggregate reference data bound".to_owned());
        }
        Ok(())
    }

    pub fn data(&self, index: usize, expected: &str) -> Result<Vec<u8>, String> {
        match self.arguments.get(index) {
            Some(Argument::Buffer { element, bytes, .. }) if element == expected => decode(bytes),
            _ => Err(format!("argument {index} is not a {expected} buffer")),
        }
    }
    pub fn launch(&self, workgroup: u32, groups: u64, arguments: usize) -> Result<(), String> {
        if self.workgroup != [workgroup, 1, 1]
            || self.grid != [u64::from(workgroup) * groups, 1, 1]
            || self.arguments.len() != arguments
        {
            return Err("reference launch/argument contract differs".to_owned());
        }
        Ok(())
    }
    pub fn corpus_shape(&self, expected: &Request) -> Result<(), String> {
        if self.kernel != expected.kernel {
            return Err("reference kernel identity differs".to_owned());
        }
        if matches!(
            self.kernel.as_str(),
            "tiled_gemm_general_v1" | "simt_gemm_general_v1"
        ) {
            return self.bf16_gemm_shape();
        }
        self.launch(
            expected.workgroup[0],
            expected.grid[0] / u64::from(expected.workgroup[0]),
            expected.arguments.len(),
        )?;
        for (a, b) in self.arguments.iter().zip(&expected.arguments) {
            let matched = match (a, b) {
                (
                    Argument::Buffer {
                        element,
                        access,
                        alignment,
                        bytes,
                    },
                    Argument::Buffer {
                        element: e,
                        access: a,
                        alignment: l,
                        bytes: b,
                    },
                ) => element == e && access == a && alignment == l && bytes.len() == b.len(),
                (Argument::Scalar { ty, bits }, Argument::Scalar { ty: t, bits: b }) => {
                    ty == t && bits.len() == b.len()
                }
                _ => false,
            };
            if !matched {
                return Err("request differs from current corpus ABI shape".to_owned());
            }
        }
        Ok(())
    }
    // Closed ordinary BF16 oracle cases, not an arbitrary shape/stride adapter.
    fn bf16_gemm_shape(&self) -> Result<(), String> {
        self.validate()?;
        if !matches!(
            self.kernel.as_str(),
            "tiled_gemm_general_v1" | "simt_gemm_general_v1"
        ) {
            return Err("unknown BF16 reference kernel".to_owned());
        }
        let shape = [
            self.scalar(3, "u32")?,
            self.scalar(4, "u32")?,
            self.scalar(5, "u32")?,
            self.scalar(6, "u32")?,
            self.scalar(7, "u32")?,
            self.scalar(8, "u32")?,
        ];
        let (a, b, logical, guards) = match shape {
            [1, 1, 1, 1, 1, 1] => (1, 1, 1, 1),
            [1, 1, 2, 2, 1, 1] => (2, 2, 1, 1),
            [16, 16, 16, 16, 16, 16] => {
                let count = self.data(2, "f32")?.len() / 4;
                match count {
                    256 => (256, 256, 256, 0), // Existing baseline retained.
                    264 => (256, 256, 256, 8),
                    _ => return Err("unreviewed BF16 output extent".to_owned()),
                }
            }
            _ => return Err("unreviewed BF16 shape or stride".to_owned()),
        };
        let groups = if self.kernel == "simt_gemm_general_v1" {
            (logical + 63) / 64
        } else {
            1
        };
        self.launch(64, groups, 11)?;
        for (index, element, access, alignment, length) in [
            (0, "u16", "read_only", 2, a * 2),
            (1, "u16", "read_only", 2, b * 2),
            (2, "f32", "read_write", 4, (logical + guards) * 4),
        ] {
            match &self.arguments[index] {
                Argument::Buffer {
                    element: e,
                    access: m,
                    alignment: l,
                    bytes,
                } if e == element
                    && m == access
                    && *l == alignment
                    && decode(bytes)?.len() == length as usize => {}
                _ => return Err("BF16 reference buffer ABI or extent".to_owned()),
            }
        }
        for index in [9, 10] {
            if !f32::from_bits(self.scalar(index, "f32")?).is_finite() {
                return Err("nonfinite BF16 reference scale".to_owned());
            }
        }
        for index in [0, 1] {
            if self.u16s(index)?.iter().any(|bits| bits & 0x7f80 == 0x7f80) {
                return Err("nonfinite BF16 operand".to_owned());
            }
        }
        let output = self.data(2, "f32")?;
        for (index, bytes) in output.chunks_exact(4).enumerate() {
            let bits = u32::from_le_bytes(bytes.try_into().unwrap());
            if (index < logical as usize && !f32::from_bits(bits).is_finite())
                || (index >= logical as usize && bits != 0x7f80_0001)
            {
                return Err("BF16 reference logical value or canary".to_owned());
            }
        }
        Ok(())
    }
    pub fn bf16_gemm_initial(&self) -> Result<Vec<f32>, String> {
        self.bf16_gemm_shape()?;
        Ok(self
            .data(2, "f32")?
            .chunks_exact(4)
            .map(|b| f32::from_bits(u32::from_le_bytes(b.try_into().unwrap())))
            .collect())
    }
    pub fn f32s(&self, index: usize) -> Result<Vec<f32>, String> {
        self.data(index, "f32")?
            .chunks_exact(4)
            .map(|b| {
                let value = f32::from_le_bytes(b.try_into().unwrap());
                if value.is_finite() {
                    Ok(value)
                } else {
                    Err("nonfinite reference input".to_owned())
                }
            })
            .collect()
    }
    pub fn u16s(&self, index: usize) -> Result<Vec<u16>, String> {
        Ok(self
            .data(index, "u16")?
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes(b.try_into().unwrap()))
            .collect())
    }
    pub fn u32s(&self, index: usize) -> Result<Vec<u32>, String> {
        Ok(self
            .data(index, "u32")?
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect())
    }
    pub fn i32s(&self, index: usize) -> Result<Vec<i32>, String> {
        Ok(self
            .data(index, "i32")?
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
            .collect())
    }
    pub fn u64s(&self, index: usize) -> Result<Vec<u64>, String> {
        Ok(self
            .data(index, "u64")?
            .chunks_exact(8)
            .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
            .collect())
    }
    pub fn scalar(&self, index: usize, expected: &str) -> Result<u32, String> {
        match self.arguments.get(index) {
            Some(Argument::Scalar { ty, bits }) if ty == expected && width(ty)? == 4 => {
                u32::from_str_radix(bits.strip_prefix("0x").ok_or("scalar prefix")?, 16)
                    .map_err(|e| e.to_string())
            }
            _ => Err(format!("argument {index} is not a {expected} scalar")),
        }
    }
    pub fn scalar64(&self, index: usize, expected: &str) -> Result<u64, String> {
        match self.arguments.get(index) {
            Some(Argument::Scalar { ty, bits }) if ty == expected && width(ty)? == 8 => {
                u64::from_str_radix(bits.strip_prefix("0x").ok_or("scalar prefix")?, 16)
                    .map_err(|e| e.to_string())
            }
            _ => Err("expected 64-bit scalar".to_owned()),
        }
    }
}

pub fn buffer(element: &str, access: &str, bytes: &[u8]) -> Argument {
    Argument::Buffer {
        element: element.to_owned(),
        access: access.to_owned(),
        alignment: width(element).expect("static element") as u32,
        bytes: encode(bytes),
    }
}
pub fn floats(values: &[f32], access: &str) -> Argument {
    buffer(
        "f32",
        access,
        &values
            .iter()
            .flat_map(|n| n.to_le_bytes())
            .collect::<Vec<_>>(),
    )
}
pub fn scalar(value: u32) -> Argument {
    Argument::Scalar {
        ty: "u32".to_owned(),
        bits: format!("0x{value:08x}"),
    }
}
pub fn float_scalar(value: f32) -> Argument {
    Argument::Scalar {
        ty: "f32".to_owned(),
        bits: format!("0x{:08x}", value.to_bits()),
    }
}
pub fn integers(values: &[i32], access: &str) -> Argument {
    buffer(
        "i32",
        access,
        &values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )
}
pub fn unsigned(values: &[u32], access: &str) -> Argument {
    buffer(
        "u32",
        access,
        &values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )
}

#[derive(Serialize)]
pub struct Output {
    argument: usize,
    element: String,
    bytes: String,
    absolute_tolerance: f64,
    relative_tolerance: f64,
}
impl Output {
    pub fn u64s(argument: usize, values: &[u64]) -> Self {
        Self {
            argument,
            element: "u64".to_owned(),
            bytes: encode(
                &values
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            ),
            absolute_tolerance: 0.0,
            relative_tolerance: 0.0,
        }
    }
    pub fn f32s(argument: usize, values: &[f32], tolerance: f64) -> Self {
        Self {
            argument,
            element: "f32".to_owned(),
            bytes: encode(
                &values
                    .iter()
                    .flat_map(|n| n.to_le_bytes())
                    .collect::<Vec<_>>(),
            ),
            absolute_tolerance: tolerance,
            relative_tolerance: tolerance,
        }
    }
    pub fn u32s(argument: usize, values: &[u32]) -> Self {
        Self {
            argument,
            element: "u32".to_owned(),
            bytes: encode(
                &values
                    .iter()
                    .flat_map(|n| n.to_le_bytes())
                    .collect::<Vec<_>>(),
            ),
            absolute_tolerance: 0.0,
            relative_tolerance: 0.0,
        }
    }
    pub fn i32s(argument: usize, values: &[i32]) -> Self {
        Self {
            argument,
            element: "i32".to_owned(),
            bytes: encode(
                &values
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            ),
            absolute_tolerance: 0.0,
            relative_tolerance: 0.0,
        }
    }
}

#[derive(Serialize)]
pub struct Response {
    schema: &'static str,
    request_sha256: [u8; 32],
    kernel: String,
    outputs: Vec<Output>,
    authority: bool,
}
impl Response {
    pub fn new(
        request: &Request,
        request_sha256: [u8; 32],
        outputs: Vec<Output>,
    ) -> Result<Self, String> {
        for (ordinal, output) in outputs.iter().enumerate() {
            let Some(Argument::Buffer {
                element,
                access,
                bytes,
                ..
            }) = request.arguments.get(output.argument)
            else {
                return Err("reference output is not a buffer".to_owned());
            };
            if element != &output.element
                || access == "read_only"
                || bytes.len() != output.bytes.len()
                || outputs[..ordinal]
                    .iter()
                    .any(|prior| prior.argument == output.argument)
                || !(0.0..=0.001).contains(&output.absolute_tolerance)
                || !(0.0..=0.001).contains(&output.relative_tolerance)
            {
                return Err("reference output shape/policy mismatch".to_owned());
            }
        }
        if request.arguments.iter().enumerate().any(|(i, a)| {
            matches!(a,
            Argument::Buffer { access, .. } if access != "read_only")
                && !outputs.iter().any(|o| o.argument == i)
        }) {
            return Err("reference output census is incomplete".to_owned());
        }
        Ok(Self {
            schema: "fe2o3-tutorial-reference-result-v92",
            request_sha256,
            kernel: request.kernel.clone(),
            outputs,
            authority: false,
        })
    }
}
