use std::fmt;

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    Lm(String),
    Arithmetic(String),
    Steganography(String),
    Message(String),
    Model(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "IO error: {}", e),
            Error::Lm(msg) => write!(f, "Language model error: {}", msg),
            Error::Arithmetic(msg) => write!(f, "Arithmetic coding error: {}", msg),
            Error::Steganography(msg) => write!(f, "Steganography error: {}", msg),
            Error::Message(msg) => write!(f, "Message error: {}", msg),
            Error::Model(msg) => write!(f, "Model error: {}", msg),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

pub type Result<T> = std::result::Result<T, Error>;