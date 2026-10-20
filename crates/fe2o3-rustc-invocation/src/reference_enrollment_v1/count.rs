//! Allocation-free projection of the closed enrollment schema, not a generic JSON parser.
//!
//! Serde's scratch and error owners cannot be used at this boundary. A cursor
//! borrows validated JSON string spans; ordering replays each kernel span at
//! most twice, so neither storage nor traversal grows quadratically with rows.

use super::{MAX_REFERENCE_ENROLLMENT_SELECTOR_BYTES_V1, validate_summary};

const SCHEMA: &str = "invalid reference enrollment request schema";
type Result<T> = std::result::Result<T, &'static str>;

pub(super) fn project(bytes: &str) -> Result<usize> {
    let mut cursor = Cursor { rest: bytes };
    let (mut version, mut bindings) = (None, None);
    cursor.object(|cursor, key| {
        if key.is("version") && version.is_none() {
            version = Some(cursor.u16()?);
        } else if key.is("bindings") && bindings.is_none() {
            bindings = Some(cursor.bindings()?);
        } else {
            return Err(SCHEMA);
        }
        Ok(())
    })?;
    cursor.whitespace();
    if !cursor.rest.is_empty() {
        return Err(SCHEMA);
    }
    let (count, invalid, unordered) = bindings.ok_or(SCHEMA)?;
    validate_summary(version.ok_or(SCHEMA)?, count, invalid, unordered)
}

struct Cursor<'a> {
    rest: &'a str,
}

impl<'a> Cursor<'a> {
    fn whitespace(&mut self) {
        self.rest = self.rest.trim_start_matches([' ', '\n', '\r', '\t']);
    }

    fn eat(&mut self, expected: char) -> bool {
        self.whitespace();
        if let Some(rest) = self.rest.strip_prefix(expected) {
            self.rest = rest;
            true
        } else {
            false
        }
    }

    fn require(&mut self, expected: char) -> Result<()> {
        if self.eat(expected) {
            Ok(())
        } else {
            Err(SCHEMA)
        }
    }

    fn object(
        &mut self,
        mut field: impl FnMut(&mut Self, JsonString<'a>) -> Result<()>,
    ) -> Result<()> {
        self.require('{')?;
        if !self.eat('}') {
            loop {
                let key = self.string()?;
                self.require(':')?;
                field(self, key)?;
                if self.eat('}') {
                    break;
                }
                self.require(',')?;
            }
        }
        Ok(())
    }

    fn bindings(&mut self) -> Result<(usize, bool, bool)> {
        self.require('[')?;
        let (mut count, mut invalid, mut unordered) = (0usize, false, false);
        let mut previous: Option<JsonString<'a>> = None;
        if !self.eat(']') {
            loop {
                let (mut kernel, mut reference) = (None, None);
                self.object(|cursor, key| {
                    if key.is("kernel") && kernel.is_none() {
                        kernel = Some(cursor.string()?);
                    } else if key.is("reference") && reference.is_none() {
                        reference = Some(cursor.string()?);
                    } else {
                        return Err(SCHEMA);
                    }
                    Ok(())
                })?;
                let kernel = kernel.ok_or(SCHEMA)?;
                let reference = reference.ok_or(SCHEMA)?;
                count = count.checked_add(1).ok_or(SCHEMA)?;
                invalid |= !kernel.selector_valid || !reference.selector_valid;
                unordered |= previous.is_some_and(|p| !p.chars().lt(kernel.chars()));
                previous = Some(kernel);
                if self.eat(']') {
                    break;
                }
                self.require(',')?;
            }
        }
        Ok((count, invalid, unordered))
    }

    fn u16(&mut self) -> Result<u16> {
        self.whitespace();
        let start = self.rest;
        let mut value = 0u16;
        while let Some(digit @ b'0'..=b'9') = self.rest.as_bytes().first().copied() {
            value = value
                .checked_mul(10)
                .and_then(|n| n.checked_add((digit - b'0') as u16))
                .ok_or(SCHEMA)?;
            self.rest = &self.rest[1..];
        }
        let digits = start.len() - self.rest.len();
        if digits == 0 || (digits > 1 && start.starts_with('0')) {
            return Err(SCHEMA);
        }
        // Decimal/exponent/sign suffixes are rejected by the enclosing delimiter.
        Ok(value)
    }

    fn string(&mut self) -> Result<JsonString<'a>> {
        self.require('"')?;
        let start = self.rest;
        let (mut bytes, mut control) = (0usize, false);
        while let Some(ch) = self.string_char()? {
            bytes = bytes.checked_add(ch.len_utf8()).ok_or(SCHEMA)?;
            control |= ch.is_control();
        }
        Ok(JsonString {
            // Includes the closing quote, so the same decoder terminates on replay.
            encoded: &start[..start.len() - self.rest.len()],
            selector_valid: bytes != 0
                && bytes <= MAX_REFERENCE_ENROLLMENT_SELECTOR_BYTES_V1
                && !control,
        })
    }

    fn next(&mut self) -> Result<char> {
        let ch = self.rest.chars().next().ok_or(SCHEMA)?;
        self.rest = &self.rest[ch.len_utf8()..];
        Ok(ch)
    }

    fn hex4(&mut self) -> Result<u16> {
        let mut value = 0;
        for _ in 0..4 {
            value = (value << 4) | self.next()?.to_digit(16).ok_or(SCHEMA)? as u16;
        }
        Ok(value)
    }

    fn string_char(&mut self) -> Result<Option<char>> {
        let ch = match self.next()? {
            '"' => return Ok(None),
            '\\' => match self.next()? {
                '"' => '"',
                '\\' => '\\',
                '/' => '/',
                'b' => '\u{8}',
                'f' => '\u{c}',
                'n' => '\n',
                'r' => '\r',
                't' => '\t',
                'u' => {
                    let first = self.hex4()?;
                    let scalar = if (0xd800..=0xdbff).contains(&first) {
                        if self.next()? != '\\' || self.next()? != 'u' {
                            return Err(SCHEMA);
                        }
                        let second = self.hex4()?;
                        if !(0xdc00..=0xdfff).contains(&second) {
                            return Err(SCHEMA);
                        }
                        0x10000 + ((u32::from(first) - 0xd800) << 10) + u32::from(second) - 0xdc00
                    } else {
                        u32::from(first)
                    };
                    char::from_u32(scalar).ok_or(SCHEMA)?
                }
                _ => return Err(SCHEMA),
            },
            ch if ch < ' ' => return Err(SCHEMA),
            ch => ch,
        };
        Ok(Some(ch))
    }
}

#[derive(Clone, Copy)]
struct JsonString<'a> {
    encoded: &'a str,
    selector_valid: bool,
}

impl JsonString<'_> {
    fn chars(self) -> impl Iterator<Item = char> {
        // Only string() constructs this private type, after validating every
        // scalar through this same decoder. Replaying its immutable span cannot fail.
        let mut cursor = Cursor { rest: self.encoded };
        std::iter::from_fn(move || cursor.string_char().ok().flatten())
    }

    fn is(self, expected: &str) -> bool {
        self.chars().eq(expected.chars())
    }
}

#[cfg(test)]
#[path = "count_tests.rs"]
mod tests;
