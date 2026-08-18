use crate::error::{Error, Result};

/// Reads bits from a byte buffer (MSB first within each byte).
pub struct BitReader<'a> {
    data: &'a [u8],
    byte_pos: usize,
    bit_pos: usize,
}

impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        BitReader {
            data,
            byte_pos: 0,
            bit_pos: 0,
        }
    }

    /// Read a single bit (0 or 1). Returns 0 when exhausted (padded with zeros).
    /// The virtual position keeps advancing even past the data end, so
    /// `bits_consumed()` reflects how many bits the decoder has read in total
    /// (including zero-padding). This is essential for the steganography
    /// encoder to know when enough bits have been consumed.
    pub fn read_bit(&mut self) -> Result<u8> {
        let bit = if self.byte_pos < self.data.len() {
            (self.data[self.byte_pos] >> (7 - self.bit_pos)) & 1
        } else {
            0 // pad with zeros
        };
        self.bit_pos += 1;
        if self.bit_pos == 8 {
            self.bit_pos = 0;
            self.byte_pos += 1;
        }
        Ok(bit)
    }

    /// Read up to 64 bits as a u64 (MSB first).
    pub fn read_bits(&mut self, n: usize) -> Result<u64> {
        if n > 64 {
            return Err(Error::Arithmetic("Cannot read more than 64 bits at once".into()));
        }
        let mut val = 0u64;
        for _ in 0..n {
            val = (val << 1) | self.read_bit()? as u64;
        }
        Ok(val)
    }

    /// Peek at the next bit without consuming it.
    pub fn peek_bit(&self) -> Result<u8> {
        if self.byte_pos >= self.data.len() {
            return Err(Error::Arithmetic("Not enough bits in message".into()));
        }
        Ok((self.data[self.byte_pos] >> (7 - self.bit_pos)) & 1)
    }

    /// Number of bits consumed so far.
    pub fn bits_consumed(&self) -> usize {
        self.byte_pos * 8 + self.bit_pos
    }

    /// Total bits available.
    pub fn total_bits(&self) -> usize {
        self.data.len() * 8
    }

    /// Returns true if we've consumed all bits.
    pub fn is_exhausted(&self) -> bool {
        self.byte_pos >= self.data.len()
    }

    /// Remaining bits.
    pub fn remaining(&self) -> usize {
        self.total_bits().saturating_sub(self.bits_consumed())
    }
}

/// Writes bits to a byte buffer (MSB first within each byte).
pub struct BitWriter {
    data: Vec<u8>,
    byte_pos: usize,
    bit_pos: usize, // 0..8, next bit to write (0 = MSB)
}

impl Default for BitWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl BitWriter {
    pub fn new() -> Self {
        BitWriter {
            data: vec![0u8; 1],
            byte_pos: 0,
            bit_pos: 0,
        }
    }

    pub fn with_capacity(bytes: usize) -> Self {
        BitWriter {
            data: vec![0u8; bytes.max(1)],
            byte_pos: 0,
            bit_pos: 0,
        }
    }

    /// Write a single bit.
    pub fn write_bit(&mut self, bit: u8) -> Result<()> {
        if self.byte_pos >= self.data.len() {
            self.data.push(0u8);
        }
        if bit != 0 {
            self.data[self.byte_pos] |= 1u8 << (7 - self.bit_pos);
        }
        self.bit_pos += 1;
        if self.bit_pos == 8 {
            self.bit_pos = 0;
            self.byte_pos += 1;
        }
        Ok(())
    }

    /// Write multiple bits from a u64 (MSB first, up to 64 bits).
    pub fn write_bits(&mut self, val: u64, n: usize) -> Result<()> {
        if n > 64 {
            return Err(Error::Arithmetic("Cannot write more than 64 bits at once".into()));
        }
        for i in (0..n).rev() {
            self.write_bit(((val >> i) & 1) as u8)?;
        }
        Ok(())
    }

    /// Finalize and return the byte buffer (padded with zeros in the last partial byte).
    pub fn finalize(mut self) -> Vec<u8> {
        // If bit_pos != 0, the last byte is incomplete; pad with zeros (already zero)
        if self.bit_pos != 0 {
            self.byte_pos += 1;
        }
        self.data.truncate(self.byte_pos);
        self.data
    }

    /// Number of bits written so far.
    pub fn bits_written(&self) -> usize {
        self.byte_pos * 8 + self.bit_pos
    }

    /// Peek at the current byte buffer (may have a partially written last byte).
    pub fn buffer(&self) -> &[u8] {
        &self.data
    }
}

/// A message that can be read as a stream of bits and also accessed as bytes.
/// Used for the steganography message (uniform random bits).
#[derive(Clone)]
pub struct Message {
    /// Byte buffer containing the message.
    data: Vec<u8>,
    /// Total number of valid bits in the message.
    num_bits: usize,
}

impl Message {
    /// Create a message from raw bytes, using all bits.
    pub fn from_bytes(data: Vec<u8>) -> Self {
        let num_bits = data.len() * 8;
        Message { data, num_bits }
    }

    /// Create a message from raw bytes, using only the first `num_bits` bits.
    pub fn from_bits(data: Vec<u8>, num_bits: usize) -> Self {
        Message { data, num_bits }
    }

    /// Generate a random message of `num_bits` bits using the given RNG.
    pub fn random<R: rand::Rng>(rng: &mut R, num_bits: usize) -> Self {
        let num_bytes = num_bits.div_ceil(8);
        let mut data = vec![0u8; num_bytes];
        rng.fill_bytes(&mut data);
        // Zero out any unused bits in the last byte
        let used_bits = num_bits % 8;
        if used_bits != 0 {
            let mask = 0xFFu8 << (8 - used_bits);
            if let Some(last) = data.last_mut() {
                *last &= mask;
            }
        }
        Message { data, num_bits }
    }

    pub fn data(&self) -> &[u8] {
        &self.data
    }

    pub fn num_bits(&self) -> usize {
        self.num_bits
    }

    pub fn num_bytes(&self) -> usize {
        self.num_bits.div_ceil(8)
    }

    /// Read a bit at the given position (0-indexed, MSB first).
    pub fn get_bit(&self, pos: usize) -> Result<u8> {
        if pos >= self.num_bits {
            return Err(Error::Message(format!("Bit position {} out of range ({} bits)", pos, self.num_bits)));
        }
        let byte = self.data[pos / 8];
        Ok((byte >> (7 - (pos % 8))) & 1)
    }

    /// Get a reader for this message.
    pub fn reader(&self) -> BitReader<'_> {
        BitReader::new(&self.data)
    }

    /// Convert to a hex string for display.
    pub fn to_hex(&self) -> String {
        let mut s = String::new();
        for &b in &self.data {
            s.push_str(&format!("{:02x}", b));
        }
        s
    }

    /// Convert message bits to a UTF-8 string (interpreting as text if possible).
    pub fn to_utf8_lossy(&self) -> String {
        let num_bytes = self.num_bits.div_ceil(8);
        String::from_utf8_lossy(&self.data[..num_bytes.min(self.data.len())]).to_string()
    }
}

impl std::fmt::Display for Message {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Message({} bits: {}...)", self.num_bits, &self.to_hex()[..self.to_hex().len().min(16)])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bit_reader_writer() {
        let mut writer = BitWriter::new();
        writer.write_bits(0b11010110, 8).unwrap();
        writer.write_bit(1).unwrap();
        writer.write_bit(0).unwrap();
        writer.write_bit(1).unwrap();
        let bytes = writer.finalize();
        assert_eq!(bytes.len(), 2);
        assert_eq!(bytes[0], 0b11010110);
        assert_eq!(bytes[1] >> 5, 0b101);

        let mut reader = BitReader::new(&bytes);
        assert_eq!(reader.read_bits(8).unwrap(), 0b11010110);
        assert_eq!(reader.read_bit().unwrap(), 1);
        assert_eq!(reader.read_bit().unwrap(), 0);
        assert_eq!(reader.read_bit().unwrap(), 1);
    }

    #[test]
    fn test_message_random() {
        let mut rng = rand::thread_rng();
        let msg = Message::random(&mut rng, 13);
        assert_eq!(msg.num_bits(), 13);
        assert_eq!(msg.num_bytes(), 2);
        // All bits should be readable
        for i in 0..13 {
            assert!(msg.get_bit(i).is_ok());
        }
        assert!(msg.get_bit(13).is_err());
    }

    #[test]
    fn test_message_bits_roundtrip() {
        let original = vec![0b10101010u8, 0b11110000];
        let msg = Message::from_bytes(original.clone());
        let mut reader = msg.reader();
        for i in 0..16 {
            let expected = (original[i / 8] >> (7 - (i % 8))) & 1;
            assert_eq!(reader.read_bit().unwrap(), expected);
        }
    }
}