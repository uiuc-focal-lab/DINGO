// src/lib.rs
use pyo3::prelude::*;
use regex_automata::dfa::dense;
use regex_automata::dfa::Automaton;
use regex_automata::util::primitives::StateID;
use regex_automata::util::start;
use regex_automata::Anchored;
use std::collections::HashMap;
use pyo3::types::{PyDict, PyBytes};

#[pyclass]
struct RegexDFA {
    dfa: Option<dense::DFA<Vec<u32>>>,
    size: usize,
    alive_states: std::collections::HashSet<usize>,
    token_transitions: Option<HashMap<(usize, usize), Option<usize>>>,
    states: Option<Vec<usize>>,
}

#[pymethods]
impl RegexDFA {
    #[new]
    fn new() -> Self {
        RegexDFA {
            dfa: None,
            size: 0,
            alive_states: std::collections::HashSet::new(),
            token_transitions: None,
            states: None,
        }
    }

    fn initialize(&mut self, regex_str: &str) -> Result<(), PyErr> {
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
            
        // Store the DFA
        self.dfa = Some(dfa);
        
        // Compute alive states
        self.compute_alive_states()?;
        
        let cnt_states = self.count_reachable_states()?;
        println!("Reachable states: {}", cnt_states);
        Ok(())
    }
    
    fn size(&mut self) -> usize {
        // Return cached value if already calculated
        if self.size > 0 {
            println!("Reachable states: {}", self.size);
            return self.size;
        }
        
        // Calculate if not already cached
        match self.count_reachable_states() {
            Ok(cnt_states) => {
                println!("Reachable states: {}", cnt_states);
                self.size = cnt_states;
                self.size
            },
            Err(e) => {
                // Handle error appropriately, perhaps log it
                // Return 0 or some default value
                println!("Error counting reachable states: {}", e);
                0
            }
        }
    }

    fn token_transitions_size(&self) -> usize {
        // Return the size of the token transitions map
        match &self.token_transitions {
            Some(transitions) => transitions.len(),
            None => 0,
        }
    }

    fn compute_token_transitions(&mut self, _py: Python, vocab: &PyDict) -> PyResult<()> {
        // Ensure we have computed the reachable states first
        if self.states.is_none() {
            self.count_reachable_states()?;
        }
        
        // Ensure DFA is initialized
        let dfa = match &self.dfa {
            Some(dfa) => dfa,
            None => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "DFA not initialized. Call initialize() first."
            ))
        };

        println!("Computing token transitions...");
        
        // Get the states
        let states = self.states.as_ref().unwrap();
        
        // Initialize the transition map
        let mut token_transitions = HashMap::new();
        
        // For each token ID and each state, compute the next state
        for (token_bytes, token_id) in vocab.iter() {
            let token_id = token_id.extract::<usize>()?;
            let token_bytes = match token_bytes.downcast::<PyBytes>() {
                Ok(bytes) => bytes.as_bytes().to_vec(),
                Err(_) => {
                    // Try to convert to string first, then to bytes
                    let s = token_bytes.extract::<String>()?;
                    s.as_bytes().to_vec()
                }
            };
            
            for &state_id in states {
                // Convert usize to StateID
                let mut current_state = match StateID::new(state_id) {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                // Process each byte of the token
                let mut result_state = Some(state_id);
                
                for &byte in &token_bytes {
                    current_state = dfa.next_state(current_state, byte);
                    
                    if dfa.is_dead_state(current_state) {
                        result_state = None;
                        break;
                    }
                    
                    result_state = Some(current_state.as_usize());
                }
                
                // Store the resulting state
                if !result_state.is_none() {
                    token_transitions.insert((state_id, token_id), result_state);
                }
            }
        }
        
        // Store the transitions in the struct
        self.token_transitions = Some(token_transitions);
        
        println!("Computed transitions for {} tokens and {} states", vocab.len(), states.len());
        
        Ok(())
    }
    
    // Accessor method to get the computed transitions (optional)
    fn get_token_transition(&self, state: usize, token_id: usize) -> PyResult<Option<usize>> {
        match &self.token_transitions {
            Some(transitions) => {
                match transitions.get(&(state, token_id)) {
                    Some(next_state) => Ok(*next_state),
                    None => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        format!("No transition found for state {} and token {}", state, token_id)
                    ))
                }
            },
            None => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Token transitions not computed. Call compute_token_transitions() first."
            ))
        }
    }

    // New method to get all token transitions as a Python dict
    fn get_all_token_transitions(&self, py: Python) -> PyResult<Py<PyDict>> {
        match &self.token_transitions {
            Some(transitions) => {
                let dict = PyDict::new(py);
                for ((state, token_id), next_state) in transitions {
                    // Only include transitions where both source and destination states are alive
                    if self.alive_states.contains(state) && next_state.map_or(false, |ns| self.alive_states.contains(&ns)) {
                        let key = (state, token_id);
                        match next_state {
                            Some(state) => dict.set_item(key, state)?,
                            None => dict.set_item(key, py.None())?
                        }
                    }
                }
                Ok(dict.into())
            },
            None => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Token transitions not computed. Call compute_token_transitions() first."
            ))
        }
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

    fn count_reachable_states(&mut self) -> PyResult<usize> {
        // Ensure DFA is initialized
        let dfa = match &self.dfa {
            Some(dfa) => dfa,
            None => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "DFA not initialized. Call initialize() first."
            ))
        };
        
        // Start from the initial state
        let config = start::Config::new();
        let start_state = match dfa.start_state(&config) {
            Ok(state) => state,
            Err(e) => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Failed to get start state: {}", e)
            ))
        };
        
        // Use BFS to traverse the DFA
        let mut visited = std::collections::HashSet::new();
        let mut queue = std::collections::VecDeque::new();
        
        // Add the start state to the queue
        queue.push_back(start_state);
        visited.insert(start_state.as_usize());
        
        // Process all reachable states
        while let Some(current_state) = queue.pop_front() {
            // For each possible input byte (0-255)
            for byte in 0..=255u8 {
                let next_state = dfa.next_state(current_state, byte);
                
                // Skip dead states
                if dfa.is_dead_state(next_state) {
                    continue;
                }
                
                // If we haven't visited this state yet
                if visited.insert(next_state.as_usize()) {
                    queue.push_back(next_state);
                }
            }
        }
        
        // Store the states in the struct for later use
        self.states = Some(visited.iter().cloned().collect());
        
        // Return the count of reachable states
        println!("Found {} reachable states", visited.len());
        Ok(visited.len())
    }

    fn compute_alive_states(&mut self) -> PyResult<()> {
        let dfa = match &self.dfa {
            Some(dfa) => dfa,
            None => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "DFA not initialized. Call initialize() first."
            ))
        };
        
        self.alive_states.clear();
        
        // First, find all accepting states
        let mut queue = std::collections::VecDeque::new();
        
        // Get all reachable states
        let reachable_states = self.get_reachable_states()?;
        
        // Add all accepting states to the queue and mark as alive
        for &state_id in &reachable_states {
            let state = StateID::new(state_id)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("Invalid state ID: {}", e)
                ))?;
                
            let eoi_state = dfa.next_eoi_state(state);
            if dfa.is_match_state(eoi_state) {
                queue.push_back(state_id);
                self.alive_states.insert(state_id);
            }
        }
        
        // Build a reverse transition map
        let mut reverse_transitions: std::collections::HashMap<usize, Vec<usize>> = 
            std::collections::HashMap::new();
            
        for &from_id in &reachable_states {
            if dfa.is_dead_state(StateID::new(from_id).unwrap()) {
                continue;
            }
            
            let from = StateID::new(from_id).unwrap();
            
            for byte in 0..=255u8 {
                let to = dfa.next_state(from, byte);
                let to_id = to.as_usize();
                
                if dfa.is_dead_state(to) {
                    continue;
                }
                
                reverse_transitions.entry(to_id)
                    .or_insert_with(Vec::new)
                    .push(from_id);
            }
        }
        
        // Perform backward search from accepting states
        while let Some(to_id) = queue.pop_front() {
            if let Some(from_ids) = reverse_transitions.get(&to_id) {
                for &from_id in from_ids {
                    if self.alive_states.insert(from_id) {
                        // If this is a new alive state, add it to the queue
                        queue.push_back(from_id);
                    }
                }
            }
        }
        
        println!("Alive states: {} out of {}", self.alive_states.len(), reachable_states.len());
        Ok(())
    }
    
    fn get_reachable_states(&self) -> PyResult<std::collections::HashSet<usize>> {
        let dfa = match &self.dfa {
            Some(dfa) => dfa,
            None => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "DFA not initialized. Call initialize() first."
            ))
        };
        
        // Start from the initial state
        let config = start::Config::new();
        let start_state = match dfa.start_state(&config) {
            Ok(state) => state,
            Err(e) => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Failed to get start state: {}", e)
            ))
        };
        
        // Use BFS to traverse the DFA
        let mut visited = std::collections::HashSet::new();
        let mut queue = std::collections::VecDeque::new();
        
        // Add the start state to the queue
        queue.push_back(start_state);
        visited.insert(start_state.as_usize());
        
        // Process all reachable states
        while let Some(current_state) = queue.pop_front() {
            // For each possible input byte (0-255)
            for byte in 0..=255u8 {
                let next_state = dfa.next_state(current_state, byte);
                
                // Skip dead states
                if dfa.is_dead_state(next_state) {
                    continue;
                }
                
                // If we haven't visited this state yet
                if visited.insert(next_state.as_usize()) {
                    queue.push_back(next_state);
                }
            }
        }
        
        Ok(visited)
    }
    
    fn prefix_matches(&self, input: &str) -> PyResult<bool> {
        // Ensure DFA is initialized
        let dfa = match &self.dfa {
            Some(dfa) => dfa,
            None => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "DFA not initialized. Call initialize() first."
            ))
        };
        
        // Get the final state by walking the DFA
        let final_state_usize = self.walk(input)?;
        // If it's a dead state, it can't lead to a match
        if dfa.is_dead_state(StateID::new(final_state_usize).unwrap()) {
            return Ok(false);
        }
        
        // Check if the state is alive (can reach an accepting state)
        Ok(self.alive_states.contains(&final_state_usize))
    }
    
    fn get_initial_state(&self) -> PyResult<usize> {
        // Ensure DFA is initialized
        let dfa = match &self.dfa {
            Some(dfa) => dfa,
            None => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "DFA not initialized. Call initialize() first."
            ))
        };
        
        // Get the initial state
        let config = start::Config::new().anchored(Anchored::Yes);
        let start_state = match dfa.start_state(&config) {
            Ok(state) => state,
            Err(e) => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Failed to get start state: {}", e)
            ))
        };
        
        Ok(start_state.as_usize())
    }
    
    fn get_final_states(&self) -> PyResult<Vec<usize>> {
        // Ensure DFA is initialized
        let dfa = match &self.dfa {
            Some(dfa) => dfa,
            None => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "DFA not initialized. Call initialize() first."
            ))
        };
        
        let mut final_states = Vec::new();
        
        // Ensure we have computed the reachable states
        if self.states.is_none() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "States not computed. Call count_reachable_states() first."
            ));
        }
        
        // Check each reachable state to see if it's a match state
        for &state_id in self.states.as_ref().unwrap() {
            let state = match StateID::new(state_id) {
                Ok(s) => s,
                Err(_) => continue,
            };
            
            let eoi_state = dfa.next_eoi_state(state);
            if dfa.is_match_state(eoi_state) {
                final_states.push(state_id);
            }
        }
        
        Ok(final_states)
    }
}

#[pymodule]
fn rust_dfa(_py: Python, m: &PyModule) -> PyResult<()> {  // This must match the module name
    m.add_class::<RegexDFA>()?;
    Ok(())
}