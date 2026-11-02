//! ``Branch``: a detached (length, gamma, parent label) value type.

use pyo3::prelude::*;

use crate::util::{py_float_repr, warn};

#[pyclass(name = "Branch", module = "phynetpy.Network")]
#[derive(Clone)]
pub struct Branch {
    length: Option<f64>,
    gamma: Option<f64>,
    parent_id: Option<String>,
}

impl Branch {
    pub fn make(length: Option<f64>, gamma: Option<f64>, parent_id: Option<String>) -> Self {
        Branch {
            length,
            gamma,
            parent_id,
        }
    }
}

#[pymethods]
impl Branch {
    #[new]
    #[pyo3(signature = (length=None, inheritance_probability=None, parent_id=None))]
    fn py_new(
        py: Python<'_>,
        length: Option<f64>,
        inheritance_probability: Option<f64>,
        parent_id: Option<String>,
    ) -> PyResult<Self> {
        let mut b = Branch {
            length: None,
            gamma: None,
            parent_id,
        };
        b.set_length(py, length)?;
        b.set_inheritance_probability(py, inheritance_probability)?;
        Ok(b)
    }

    #[getter]
    fn get_length(&self) -> Option<f64> {
        self.length
    }

    #[setter]
    fn set_length(&mut self, py: Python<'_>, value: Option<f64>) -> PyResult<()> {
        if let Some(v) = value {
            if v < 0.0 {
                warn(
                    py,
                    "Branch length cannot be negative, the length will not be changed",
                )?;
                return Ok(());
            }
        }
        self.length = value;
        Ok(())
    }

    #[getter]
    fn get_inheritance_probability(&self) -> Option<f64> {
        self.gamma
    }

    #[setter]
    fn set_inheritance_probability(&mut self, py: Python<'_>, value: Option<f64>) -> PyResult<()> {
        if let Some(v) = value {
            if !(0.0..=1.0).contains(&v) {
                warn(
                    py,
                    "Inheritance probability must be between 0 and 1, the inheritance \
                     probability will not be changed",
                )?;
                return Ok(());
            }
        }
        self.gamma = value;
        Ok(())
    }

    #[getter]
    fn get_parent_id(&self) -> Option<String> {
        self.parent_id.clone()
    }

    #[setter]
    fn set_parent_id(&mut self, value: Option<String>) {
        self.parent_id = value;
    }

    fn __str__(&self, py: Python<'_>) -> PyResult<String> {
        let l = match self.length {
            Some(v) => py_float_repr(py, v)?,
            None => "None".to_string(),
        };
        let g = match self.gamma {
            Some(v) => py_float_repr(py, v)?,
            None => "None".to_string(),
        };
        Ok(format!("Branch(length={l}, inheritance_probability={g})"))
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        self.__str__(py)
    }

    fn __getnewargs__(&self) -> (Option<f64>, Option<f64>, Option<String>) {
        (self.length, self.gamma, self.parent_id.clone())
    }
}
