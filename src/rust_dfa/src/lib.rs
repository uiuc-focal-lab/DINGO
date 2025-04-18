// src/lib.rs
use pyo3::prelude::*;
use regex_automata::dfa::dense;
use regex_automata::dfa::Automaton;
use regex_automata::util::primitives::StateID;
use regex_automata::util::start;
use regex_automata::Anchored;

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

    fn walk(&self, input: &str) -> PyResult<usize> {
        // Ensure DFA is initialized
        let dfa = match &self.dfa {
            Some(dfa) => dfa,
            None => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "DFA not initialized. Call initialize() first."
            ))
        };
        
        // Convert string to bytes
        let bytes = input.as_bytes();
        
        // Set up the config with anchored mode
        let config = start::Config::new().anchored(Anchored::Yes);
        
        // Get the start state
        let mut state = match dfa.start_state(&config) {
            Ok(state) => state,
            Err(e) => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Failed to get start state: {}", e)
            ))
        };
        
        // Walk through the DFA state by state
        for &byte in bytes {
            state = dfa.next_state(state, byte);
            
            if dfa.is_dead_state(state) {
                break;
            }
        }
        
        // Return the state ID as a usize
        Ok(state.as_usize())
    }
    
    fn matches(&self, input: &str) -> PyResult<bool> {
        // Ensure DFA is initialized
        let dfa = match &self.dfa {
            Some(dfa) => dfa,
            None => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "DFA not initialized. Call initialize() first."
            ))
        };
        
        // Get the final state by walking the DFA
        let final_state_usize = self.walk(input)?;
        
        // Convert usize to StateID
        let final_state = StateID::new(final_state_usize)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Invalid state ID: {}", e)
            ))?;
        
        // Check if the final state is a match state after EOI
        let eoi_state = dfa.next_eoi_state(final_state);
        
        // Return whether it's a match state
        Ok(dfa.is_match_state(eoi_state))
    }
}

#[pymodule]
fn rust_dfa(_py: Python, m: &PyModule) -> PyResult<()> {  // This must match the module name
    m.add_class::<RegexDFA>()?;
    Ok(())
}