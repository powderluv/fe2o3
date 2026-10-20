//! Closed schema: no dynamic key owners, generic value tree, or positional structs.
use super::*;
use serde::Deserializer;
use serde::de::{DeserializeSeed, Error, MapAccess, SeqAccess, Visitor};

const SCHEMA: &str = "invalid reference enrollment request schema";

pub(super) struct Parsed {
    pub request: ReferenceEnrollmentRequestV1,
    pub binding_count: usize,
    pub invalid_selector: bool,
}

pub(super) fn request(bytes: &str) -> Result<Parsed, serde_json::Error> {
    let mut decoder = serde_json::Deserializer::from_str(bytes);
    let parsed = decoder.deserialize_map(Request)?;
    decoder.end()?;
    Ok(parsed)
}

enum Field {
    Version,
    Bindings,
    Kernel,
    Reference,
}
struct Key;
impl<'de> DeserializeSeed<'de> for Key {
    type Value = Field;
    fn deserialize<D: Deserializer<'de>>(self, decoder: D) -> Result<Field, D::Error> {
        decoder.deserialize_identifier(self)
    }
}
impl<'de> Visitor<'de> for Key {
    type Value = Field;
    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("an enrollment field name")
    }
    fn visit_str<E: Error>(self, key: &str) -> Result<Field, E> {
        match key {
            "version" => Ok(Field::Version),
            "bindings" => Ok(Field::Bindings),
            "kernel" => Ok(Field::Kernel),
            "reference" => Ok(Field::Reference),
            _ => Err(E::custom(SCHEMA)),
        }
    }
}

struct Request;
impl<'de> Visitor<'de> for Request {
    type Value = Parsed;
    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("an enrollment request object")
    }
    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Parsed, A::Error> {
        let (mut version, mut bindings) = (None, None);
        while let Some(field) = map.next_key_seed(Key)? {
            match field {
                Field::Version if version.is_none() => version = Some(map.next_value::<u16>()?),
                Field::Bindings if bindings.is_none() => {
                    bindings = Some(map.next_value_seed(Bindings)?)
                }
                _ => return Err(A::Error::custom(SCHEMA)),
            }
        }
        let version = version.ok_or_else(|| A::Error::custom(SCHEMA))?;
        let bindings = bindings.ok_or_else(|| A::Error::custom(SCHEMA))?;
        Ok(Parsed {
            request: ReferenceEnrollmentRequestV1 {
                version,
                bindings: bindings.rows,
            },
            binding_count: bindings.count,
            invalid_selector: bindings.invalid_selector,
        })
    }
}

struct ParsedBindings {
    rows: Vec<ReferenceEnrollmentBindingV1>,
    count: usize,
    invalid_selector: bool,
}
struct Bindings;
impl<'de> DeserializeSeed<'de> for Bindings {
    type Value = ParsedBindings;
    fn deserialize<D: Deserializer<'de>>(self, decoder: D) -> Result<Self::Value, D::Error> {
        decoder.deserialize_seq(self)
    }
}
impl<'de> Visitor<'de> for Bindings {
    type Value = ParsedBindings;
    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("an enrollment bindings array")
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut sequence: A) -> Result<Self::Value, A::Error> {
        let mut result = ParsedBindings {
            rows: Vec::new(),
            count: 0,
            invalid_selector: false,
        };
        result
            .rows
            .try_reserve_exact(MAX_REFERENCE_ENROLLMENT_BINDINGS_V1)
            .map_err(|_| A::Error::custom(SCHEMA))?;
        while let Some((row, valid)) = sequence.next_element_seed(Binding {
            retain: result.count < MAX_REFERENCE_ENROLLMENT_BINDINGS_V1,
        })? {
            result.count = result
                .count
                .checked_add(1)
                .ok_or_else(|| A::Error::custom(SCHEMA))?;
            result.invalid_selector |= !valid;
            if result.count <= MAX_REFERENCE_ENROLLMENT_BINDINGS_V1 {
                result.rows.push(row);
            }
        }
        Ok(result)
    }
}

struct Binding {
    retain: bool,
}
impl<'de> DeserializeSeed<'de> for Binding {
    type Value = (ReferenceEnrollmentBindingV1, bool);
    fn deserialize<D: Deserializer<'de>>(self, decoder: D) -> Result<Self::Value, D::Error> {
        decoder.deserialize_map(self)
    }
}
impl<'de> Visitor<'de> for Binding {
    type Value = (ReferenceEnrollmentBindingV1, bool);
    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("an enrollment binding object")
    }
    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
        let (mut kernel, mut reference) = (None, None);
        while let Some(field) = map.next_key_seed(Key)? {
            match field {
                Field::Kernel if kernel.is_none() => {
                    kernel = Some(map.next_value_seed(Selector {
                        retain: self.retain,
                    })?)
                }
                Field::Reference if reference.is_none() => {
                    reference = Some(map.next_value_seed(Selector {
                        retain: self.retain,
                    })?)
                }
                _ => return Err(A::Error::custom(SCHEMA)),
            }
        }
        let (kernel, valid_kernel) = kernel.ok_or_else(|| A::Error::custom(SCHEMA))?;
        let (reference, valid_reference) = reference.ok_or_else(|| A::Error::custom(SCHEMA))?;
        Ok((
            ReferenceEnrollmentBindingV1 { kernel, reference },
            valid_kernel && valid_reference,
        ))
    }
}

struct Selector {
    retain: bool,
}
impl<'de> DeserializeSeed<'de> for Selector {
    type Value = (String, bool);
    fn deserialize<D: Deserializer<'de>>(self, decoder: D) -> Result<Self::Value, D::Error> {
        decoder.deserialize_str(self)
    }
}
impl<'de> Visitor<'de> for Selector {
    type Value = (String, bool);
    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a string")
    }
    fn visit_str<E: Error>(self, value: &str) -> Result<Self::Value, E> {
        let valid = !value.is_empty()
            && value.len() <= MAX_REFERENCE_ENROLLMENT_SELECTOR_BYTES_V1
            && !value.chars().any(char::is_control);
        let mut retained = String::new();
        if self.retain && valid {
            retained
                .try_reserve_exact(value.len())
                .map_err(|_| E::custom(SCHEMA))?;
            retained.push_str(value);
        }
        // Defer semantic refusals until the entire JSON schema has been checked.
        Ok((retained, valid))
    }
}
