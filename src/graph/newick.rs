//! Extended-Newick parsing into a ``Network`` (``Network.from_newick``).

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyType};
use rustc_hash::{FxHashMap, FxHashSet};

use super::edge::Edge;
use super::network::Network;
use super::node::Node;
use crate::util::{is_newick_error, newick_error, node_error};

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Open,
    Close,
    Comma,
    Colon,
    /// The text following a ':' (possibly empty).
    Len(String),
    Comment(String),
    Name(String),
}

impl Tok {
    /// The reference parser tests ``token not in ':,);['`` on raw strings; a
    /// single structural character (or an empty string) is "in" that set.
    fn is_structural(&self) -> bool {
        match self {
            Tok::Open | Tok::Close | Tok::Comma | Tok::Colon => true,
            Tok::Len(s) | Tok::Name(s) | Tok::Comment(s) => {
                s.is_empty() || (s.len() == 1 && ":,);[".contains(s.as_str()))
            }
        }
    }

    fn text(&self) -> String {
        match self {
            Tok::Open => "(".into(),
            Tok::Close => ")".into(),
            Tok::Comma => ",".into(),
            Tok::Colon => ":".into(),
            Tok::Len(s) | Tok::Name(s) | Tok::Comment(s) => s.clone(),
        }
    }
}

fn tokenize(py: Python<'_>, s: &str) -> PyResult<Vec<Tok>> {
    let chars: Vec<char> = s.chars().collect();
    let n = chars.len();
    let mut tokens = Vec::new();
    let mut i = 0;
    while i < n {
        let c = chars[i];
        match c {
            '(' => {
                tokens.push(Tok::Open);
                i += 1;
            }
            ')' => {
                tokens.push(Tok::Close);
                i += 1;
            }
            ',' => {
                tokens.push(Tok::Comma);
                i += 1;
            }
            ';' => {
                tokens.push(Tok::Name(";".into()));
                i += 1;
            }
            ':' => {
                tokens.push(Tok::Colon);
                i += 1;
                let mut j = i;
                while j < n && !"(),;:[".contains(chars[j]) {
                    j += 1;
                }
                tokens.push(Tok::Len(chars[i..j].iter().collect()));
                i = j;
            }
            '[' => {
                let mut j = i;
                while j < n && chars[j] != ']' {
                    j += 1;
                }
                if j >= n {
                    return Err(newick_error(py, "Unclosed comment block"));
                }
                tokens.push(Tok::Comment(chars[i..=j].iter().collect()));
                i = j + 1;
            }
            c if c.is_whitespace() => {
                i += 1;
            }
            _ => {
                let mut j = i;
                while j < n && !"(),;:[]".contains(chars[j]) && !chars[j].is_whitespace() {
                    j += 1;
                }
                if i < j {
                    tokens.push(Tok::Name(chars[i..j].iter().collect()));
                }
                i = j;
            }
        }
    }
    Ok(tokens)
}

struct PNode {
    name: Option<String>,
    length: f64,
    children: Vec<PNode>,
    gamma: Option<f64>,
}

fn py_float(s: &str) -> Option<f64> {
    let t = s.trim();
    if t.is_empty() {
        return None;
    }
    let lower = t.to_ascii_lowercase();
    match lower.as_str() {
        "inf" | "+inf" | "infinity" | "+infinity" => return Some(f64::INFINITY),
        "-inf" | "-infinity" => return Some(f64::NEG_INFINITY),
        "nan" | "+nan" | "-nan" => return Some(f64::NAN),
        _ => {}
    }
    let cleaned: String = t.chars().filter(|&c| c != '_').collect();
    cleaned.parse::<f64>().ok()
}

fn parse_comment(tok: &str) -> Option<f64> {
    if tok.starts_with('[') && tok.ends_with(']') && tok.len() >= 2 {
        let content = &tok[1..tok.len() - 1];
        if let Some(rest) = content.strip_prefix("&gamma=") {
            return py_float(rest);
        }
    }
    None
}

fn parse_tokens(py: Python<'_>, tokens: &[Tok], mut idx: usize) -> PyResult<(PNode, usize)> {
    if idx >= tokens.len() {
        return Err(newick_error(py, "Unexpected end of string"));
    }
    let mut node = PNode {
        name: None,
        length: 1.0,
        children: Vec::new(),
        gamma: None,
    };
    if tokens[idx] == Tok::Open {
        idx += 1;
        loop {
            let (child, next) = parse_tokens(py, tokens, idx)?;
            idx = next;
            node.children.push(child);
            if idx >= tokens.len() {
                return Err(newick_error(py, "Unexpected end of string"));
            }
            match tokens[idx] {
                Tok::Comma => idx += 1,
                Tok::Close => {
                    idx += 1;
                    break;
                }
                ref t => {
                    return Err(newick_error(
                        py,
                        format!("Unexpected token: {}", t.text()),
                    ))
                }
            }
        }
        if idx < tokens.len() && !tokens[idx].is_structural() {
            node.name = Some(tokens[idx].text());
            idx += 1;
        }
    } else if !tokens[idx].is_structural() {
        node.name = Some(tokens[idx].text());
        idx += 1;
    }

    if idx < tokens.len() {
        if let Tok::Comment(c) = &tokens[idx] {
            node.gamma = parse_comment(c);
            idx += 1;
        }
    }
    if idx < tokens.len() && tokens[idx] == Tok::Colon {
        idx += 1;
        if idx >= tokens.len() {
            return Err(newick_error(py, "Expected branch length after ':'"));
        }
        let text = tokens[idx].text();
        match py_float(&text) {
            Some(v) => node.length = v,
            None => {
                return Err(newick_error(
                    py,
                    format!("Invalid branch length: {text}"),
                ))
            }
        }
        idx += 1;
    }
    if idx < tokens.len() {
        if let Tok::Comment(c) = &tokens[idx] {
            if node.gamma.is_none() {
                node.gamma = parse_comment(c);
            }
            idx += 1;
        }
    }
    Ok((node, idx))
}

struct Builder<'a, 'py> {
    py: Python<'py>,
    net: &'a Bound<'py, Network>,
    node_map: FxHashMap<String, Py<Node>>,
    explicit: FxHashSet<u64>,
}

impl<'a, 'py> Builder<'a, 'py> {
    fn build(&mut self, pn: &PNode, parent: Option<&Py<Node>>, time: f64) -> PyResult<()> {
        let py = self.py;
        let name = match &pn.name {
            Some(n) if !n.is_empty() => n.clone(),
            _ => format!("Internal{}", self.node_map.len()),
        };
        let node: Py<Node> = match self.node_map.get(&name) {
            Some(n) => n.clone_ref(py),
            None => {
                let is_retic = name.starts_with('#');
                let n = Py::new(py, Node::make(py, name.clone(), is_retic, None, None, None))?;
                if time < 0.0 {
                    return Err(node_error(
                        py,
                        "Please set speciation time, t, to a non-negative number!",
                    ));
                }
                n.borrow_mut(py).t = Some(time);
                self.net
                    .borrow_mut()
                    .add_node(self.net.as_any(), n.bind(py))?;
                self.node_map.insert(name, n.clone_ref(py));
                n
            }
        };
        if let Some(p) = parent {
            let edge = Py::new(
                py,
                Edge::make(
                    py,
                    p.clone_ref(py),
                    node.clone_ref(py),
                    Some(pn.length),
                    Some(0.0),
                    Some(0.0),
                    Some("no tag assigned yet".to_string()),
                ),
            )?;
            if let Some(g) = pn.gamma {
                Edge::set_gamma(edge.bind(py), g)?;
                self.explicit.insert(edge.borrow(py).uid);
            }
            self.net.borrow_mut().add_edge(self.net.as_any(), edge.bind(py))?;
        }
        for child in &pn.children {
            self.build(child, Some(&node), time + child.length)?;
        }
        Ok(())
    }
}

/// Fill in inheritance probabilities a Newick string left implicit.
pub fn complete_hybrid_gammas(py: Python<'_>, net: &Network, explicit: &FxHashSet<u64>) -> PyResult<()> {
    for (_, entry) in net.nodes.iter() {
        if entry.in_edges.len() < 2 {
            continue;
        }
        let edges: Vec<u64> = entry.in_edges.iter().copied().collect();
        let unknown: Vec<u64> = edges.iter().copied().filter(|e| !explicit.contains(e)).collect();
        if unknown.is_empty() {
            continue;
        }
        let known: f64 = edges
            .iter()
            .filter(|e| explicit.contains(e))
            .map(|e| net.edges[e].obj.borrow(py).gamma.unwrap_or(0.0))
            .sum();
        let remainder = (1.0 - known).clamp(0.0, 1.0);
        let share = remainder / unknown.len() as f64;
        for e in unknown {
            Edge::set_gamma(net.edges[&e].obj.bind(py), share)?;
        }
    }
    Ok(())
}

fn parse_into(
    py: Python<'_>,
    cls: &Bound<'_, PyType>,
    newick: &str,
    unit: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let s = newick.trim();
    if !s.ends_with(';') {
        return Err(newick_error(py, "Newick string must end with semicolon"));
    }
    let s = &s[..s.len() - 1];
    let tokens = tokenize(py, s)?;
    if tokens.is_empty() {
        return Err(newick_error(py, "Empty Newick string"));
    }
    let (root, _) = parse_tokens(py, &tokens, 0)?;

    let kwargs = PyDict::new(py);
    kwargs.set_item("branch_length_unit", unit)?;
    let obj = cls.call((), Some(&kwargs))?;
    let net = obj.cast::<Network>()?.clone();
    let explicit = {
        let mut b = Builder {
            py,
            net: &net,
            node_map: FxHashMap::default(),
            explicit: FxHashSet::default(),
        };
        b.build(&root, None, 0.0)?;
        b.explicit
    };
    complete_hybrid_gammas(py, &net.borrow(), &explicit)?;
    Network::clean_impl(&net, [true, true, true])?;
    Ok(obj.unbind())
}

/// ``Network.from_newick``: parse ``newick`` into a new instance of ``cls``.
#[pyfunction]
#[pyo3(signature = (cls, newick, branch_length_unit=None))]
pub fn parse_newick(
    py: Python<'_>,
    cls: Bound<'_, PyType>,
    newick: &str,
    branch_length_unit: Option<Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    match parse_into(py, &cls, newick, branch_length_unit.as_ref()) {
        Ok(v) => Ok(v),
        Err(e) if is_newick_error(py, &e) => Err(e),
        Err(e) => {
            let msg = e
                .value(py)
                .str()
                .map(|s| s.to_string())
                .unwrap_or_else(|_| e.to_string());
            Err(newick_error(py, format!("Failed to parse Newick string: {msg}")))
        }
    }
}


/// ``Network._complete_hybrid_gammas(network, explicit_edges)``: fill in the
/// inheritance probabilities a Newick string left implicit.
#[pyfunction]
#[pyo3(name = "_complete_hybrid_gammas")]
pub fn complete_hybrid_gammas_py(
    py: Python<'_>,
    network: &Bound<'_, Network>,
    explicit: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let mut ids: FxHashSet<u64> = FxHashSet::default();
    for item in explicit.try_iter()? {
        let item = item?;
        if let Ok(e) = item.cast::<Edge>() {
            ids.insert(e.borrow().uid);
        }
    }
    let net = network.borrow();
    complete_hybrid_gammas(py, &net, &ids)
}
