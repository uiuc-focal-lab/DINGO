// src/lib.rs
use pyo3::prelude::*;
use regex_automata::dfa::dense;

#[pyclass]
struct RegexDFA {
    dfa: Option<dense::DFA<Vec<u32>>>,
    size: usize,
}

#[pymethods]
impl RegexDFA {
    #[new]
    fn new() -> Self {
        RegexDFA {
            dfa: None,
            size: 0,
        }
    }

    fn initialize(&mut self, regex_str: &str) -> PyResult<usize> {
        // Create a DFA directly from the regex with minimization enabled
        let dfa = match dense::Builder::new()
            .configure(dense::Config::new().minimize(true))
            .build(regex_str) {
                Ok(dfa) => dfa,
                Err(e) => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("Failed to build DFA: {}", e)
                ))
            };
        
        // print the regex and the size of the DFA
        println!("Regex: {}", regex_str);
        println!("DFA size: {}", dfa.memory_usage());

        self.size = dfa.memory_usage();
        
        // Store the DFA
        self.dfa = Some(dfa);
        
        Ok(self.size)
    }
    
    fn get_size(&self) -> usize {
        self.size
    }
}

#[pymodule]
fn rust_dfa(_py: Python, m: &PyModule) -> PyResult<()> {  // This must match the module name
    m.add_class::<RegexDFA>()?;
    Ok(())
}