//! Conjunctive and Disjunctive Normal Forms

use crate::formula::{Formula, Variable};
use std::collections::HashSet;
use std::fmt;

/// A literal (variable or its negation)
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Literal {
    pub variable: Variable,
    pub negated: bool,
}

impl Literal {
    pub fn new(variable: Variable, negated: bool) -> Self {
        Literal { variable, negated }
    }

    pub fn positive(variable: Variable) -> Self {
        Literal {
            variable,
            negated: false,
        }
    }

    pub fn negative(variable: Variable) -> Self {
        Literal {
            variable,
            negated: true,
        }
    }

    pub fn negate(&self) -> Self {
        Literal {
            variable: self.variable.clone(),
            negated: !self.negated,
        }
    }

    pub fn to_formula(&self) -> Formula {
        let var = Formula::Var(self.variable.clone());
        if self.negated {
            Formula::Not(Box::new(var))
        } else {
            var
        }
    }
}

impl fmt::Display for Literal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.negated {
            write!(f, "¬{}", self.variable)
        } else {
            write!(f, "{}", self.variable)
        }
    }
}

/// A clause (disjunction of literals)
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Clause {
    pub literals: Vec<Literal>,
}

impl Clause {
    pub fn new(literals: Vec<Literal>) -> Self {
        Clause { literals }
    }

    pub fn empty() -> Self {
        Clause {
            literals: Vec::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.literals.is_empty()
    }

    pub fn is_unit(&self) -> bool {
        self.literals.len() == 1
    }

    pub fn contains_variable(&self, var: &Variable) -> bool {
        self.literals.iter().any(|lit| &lit.variable == var)
    }

    pub fn is_tautology(&self) -> bool {
        // A clause is a tautology if it contains both a variable and its negation
        let mut seen = HashSet::new();
        for lit in &self.literals {
            if seen.contains(&lit.negate()) {
                return true;
            }
            seen.insert(lit.clone());
        }
        false
    }

    pub fn to_formula(&self) -> Formula {
        if self.literals.is_empty() {
            return Formula::False;
        }
        if self.literals.len() == 1 {
            return self.literals[0].to_formula();
        }
        Formula::Or(self.literals.iter().map(|lit| lit.to_formula()).collect())
    }
}

impl fmt::Display for Clause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.literals.is_empty() {
            write!(f, "⊥")
        } else {
            write!(f, "(")?;
            for (i, lit) in self.literals.iter().enumerate() {
                if i > 0 {
                    write!(f, " ∨ ")?;
                }
                write!(f, "{}", lit)?;
            }
            write!(f, ")")
        }
    }
}

/// Conjunctive Normal Form (CNF) - conjunction of clauses
#[derive(Clone, Debug, PartialEq)]
pub struct Cnf {
    pub clauses: Vec<Clause>,
}

impl Cnf {
    pub fn new(clauses: Vec<Clause>) -> Self {
        Cnf { clauses }
    }

    pub fn empty() -> Self {
        Cnf {
            clauses: Vec::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.clauses.is_empty()
    }

    pub fn add_clause(&mut self, clause: Clause) {
        self.clauses.push(clause);
    }

    pub fn to_formula(&self) -> Formula {
        if self.clauses.is_empty() {
            return Formula::True;
        }
        if self.clauses.len() == 1 {
            return self.clauses[0].to_formula();
        }
        Formula::And(self.clauses.iter().map(|c| c.to_formula()).collect())
    }

    /// Convert a formula to CNF using Tseitin transformation
    pub fn from_formula(formula: &Formula) -> Self {
        // First eliminate implications and equivalences
        let simplified = eliminate_implications(formula);
        // Then push negations inward (De Morgan's laws)
        let nnf = to_nnf(&simplified);
        // Finally distribute OR over AND
        to_cnf(&nnf)
    }
}

impl fmt::Display for Cnf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.clauses.is_empty() {
            write!(f, "⊤")
        } else {
            for (i, clause) in self.clauses.iter().enumerate() {
                if i > 0 {
                    write!(f, " ∧ ")?;
                }
                write!(f, "{}", clause)?;
            }
            Ok(())
        }
    }
}

/// Disjunctive Normal Form (DNF) - disjunction of conjunctions
#[derive(Clone, Debug, PartialEq)]
pub struct Dnf {
    pub terms: Vec<Vec<Literal>>,
}

impl Dnf {
    pub fn new(terms: Vec<Vec<Literal>>) -> Self {
        Dnf { terms }
    }

    pub fn empty() -> Self {
        Dnf { terms: Vec::new() }
    }

    pub fn to_formula(&self) -> Formula {
        if self.terms.is_empty() {
            return Formula::False;
        }
        if self.terms.len() == 1 {
            let term = &self.terms[0];
            if term.is_empty() {
                return Formula::True;
            }
            if term.len() == 1 {
                return term[0].to_formula();
            }
            return Formula::And(term.iter().map(|lit| lit.to_formula()).collect());
        }
        Formula::Or(
            self.terms
                .iter()
                .map(|term| {
                    if term.is_empty() {
                        Formula::True
                    } else if term.len() == 1 {
                        term[0].to_formula()
                    } else {
                        Formula::And(term.iter().map(|lit| lit.to_formula()).collect())
                    }
                })
                .collect(),
        )
    }

    /// Convert a formula to DNF
    pub fn from_formula(formula: &Formula) -> Self {
        let simplified = eliminate_implications(formula);
        let nnf = to_nnf(&simplified);
        to_dnf(&nnf)
    }
}

impl fmt::Display for Dnf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.terms.is_empty() {
            write!(f, "⊥")
        } else {
            for (i, term) in self.terms.iter().enumerate() {
                if i > 0 {
                    write!(f, " ∨ ")?;
                }
                write!(f, "(")?;
                for (j, lit) in term.iter().enumerate() {
                    if j > 0 {
                        write!(f, " ∧ ")?;
                    }
                    write!(f, "{}", lit)?;
                }
                write!(f, ")")?;
            }
            Ok(())
        }
    }
}

/// Eliminate implications and equivalences
fn eliminate_implications(formula: &Formula) -> Formula {
    match formula {
        Formula::True | Formula::False | Formula::Var(_) => formula.clone(),
        Formula::Not(f) => Formula::Not(Box::new(eliminate_implications(f))),
        Formula::And(fs) => {
            Formula::And(fs.iter().map(eliminate_implications).collect())
        }
        Formula::Or(fs) => Formula::Or(fs.iter().map(eliminate_implications).collect()),
        // A → B becomes ¬A ∨ B
        Formula::Implies(left, right) => Formula::Or(vec![
            Formula::Not(Box::new(eliminate_implications(left))),
            eliminate_implications(right),
        ]),
        // A ↔ B becomes (¬A ∨ B) ∧ (A ∨ ¬B)
        Formula::Iff(left, right) => {
            let l = eliminate_implications(left);
            let r = eliminate_implications(right);
            Formula::And(vec![
                Formula::Or(vec![Formula::Not(Box::new(l.clone())), r.clone()]),
                Formula::Or(vec![l, Formula::Not(Box::new(r))]),
            ])
        }
        // A ⊕ B becomes (A ∨ B) ∧ (¬A ∨ ¬B)
        Formula::Xor(left, right) => {
            let l = eliminate_implications(left);
            let r = eliminate_implications(right);
            Formula::And(vec![
                Formula::Or(vec![l.clone(), r.clone()]),
                Formula::Or(vec![
                    Formula::Not(Box::new(l)),
                    Formula::Not(Box::new(r)),
                ]),
            ])
        }
    }
}

/// Convert to Negation Normal Form (NNF)
fn to_nnf(formula: &Formula) -> Formula {
    match formula {
        Formula::True | Formula::False | Formula::Var(_) => formula.clone(),
        Formula::Not(f) => match f.as_ref() {
            Formula::True => Formula::False,
            Formula::False => Formula::True,
            Formula::Var(_) => formula.clone(),
            Formula::Not(inner) => to_nnf(inner),
            Formula::And(fs) => {
                Formula::Or(fs.iter().map(|f| to_nnf(&Formula::Not(Box::new(f.clone())))).collect())
            }
            Formula::Or(fs) => {
                Formula::And(fs.iter().map(|f| to_nnf(&Formula::Not(Box::new(f.clone())))).collect())
            }
            _ => panic!("Implications should be eliminated first"),
        },
        Formula::And(fs) => Formula::And(fs.iter().map(to_nnf).collect()),
        Formula::Or(fs) => Formula::Or(fs.iter().map(to_nnf).collect()),
        _ => panic!("Implications should be eliminated first"),
    }
}

/// Convert NNF to CNF
fn to_cnf(formula: &Formula) -> Cnf {
    match formula {
        Formula::True => Cnf::empty(),
        Formula::False => {
            let mut cnf = Cnf::empty();
            cnf.add_clause(Clause::empty());
            cnf
        }
        Formula::Var(v) => {
            let mut cnf = Cnf::empty();
            cnf.add_clause(Clause::new(vec![Literal::positive(v.clone())]));
            cnf
        }
        Formula::Not(f) => {
            if let Formula::Var(v) = f.as_ref() {
                let mut cnf = Cnf::empty();
                cnf.add_clause(Clause::new(vec![Literal::negative(v.clone())]));
                cnf
            } else {
                panic!("Should be in NNF")
            }
        }
        Formula::And(fs) => {
            let mut cnf = Cnf::empty();
            for f in fs {
                let sub_cnf = to_cnf(f);
                cnf.clauses.extend(sub_cnf.clauses);
            }
            cnf
        }
        Formula::Or(fs) => {
            // Distribute OR over AND
            let mut result = vec![Vec::new()];
            for f in fs {
                let cnf = to_cnf(f);
                let mut new_result = Vec::new();
                for existing in &result {
                    for clause in &cnf.clauses {
                        let mut combined = existing.clone();
                        combined.extend(clause.literals.clone());
                        new_result.push(combined);
                    }
                }
                result = new_result;
            }
            let mut cnf = Cnf::empty();
            for literals in result {
                cnf.add_clause(Clause::new(literals));
            }
            cnf
        }
        _ => panic!("Should be in NNF"),
    }
}

/// Convert NNF to DNF
fn to_dnf(formula: &Formula) -> Dnf {
    match formula {
        Formula::True => Dnf::new(vec![Vec::new()]),
        Formula::False => Dnf::empty(),
        Formula::Var(v) => Dnf::new(vec![vec![Literal::positive(v.clone())]]),
        Formula::Not(f) => {
            if let Formula::Var(v) = f.as_ref() {
                Dnf::new(vec![vec![Literal::negative(v.clone())]])
            } else {
                panic!("Should be in NNF")
            }
        }
        Formula::Or(fs) => {
            let mut dnf = Dnf::empty();
            for f in fs {
                let sub_dnf = to_dnf(f);
                dnf.terms.extend(sub_dnf.terms);
            }
            dnf
        }
        Formula::And(fs) => {
            // Distribute AND over OR
            let mut result = vec![Vec::new()];
            for f in fs {
                let dnf = to_dnf(f);
                let mut new_result = Vec::new();
                for existing in &result {
                    for term in &dnf.terms {
                        let mut combined = existing.clone();
                        combined.extend(term.clone());
                        new_result.push(combined);
                    }
                }
                result = new_result;
            }
            Dnf::new(result)
        }
        _ => panic!("Should be in NNF"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_literal_creation() {
        let var = Variable::new("x");
        let lit_pos = Literal::positive(var.clone());
        let lit_neg = Literal::negative(var.clone());

        assert!(!lit_pos.negated);
        assert!(lit_neg.negated);
        assert_eq!(lit_pos.negate(), lit_neg);
    }

    #[test]
    fn test_clause_tautology() {
        let var = Variable::new("x");
        let clause = Clause::new(vec![
            Literal::positive(var.clone()),
            Literal::negative(var.clone()),
        ]);
        assert!(clause.is_tautology());

        let clause2 = Clause::new(vec![Literal::positive(var.clone())]);
        assert!(!clause2.is_tautology());
    }

    #[test]
    fn test_cnf_from_simple_formula() {
        // x ∧ y
        let formula = Formula::and(vec![Formula::var("x"), Formula::var("y")]);
        let cnf = Cnf::from_formula(&formula);

        assert_eq!(cnf.clauses.len(), 2);
    }

    #[test]
    fn test_cnf_from_or_formula() {
        // x ∨ y
        let formula = Formula::or(vec![Formula::var("x"), Formula::var("y")]);
        let cnf = Cnf::from_formula(&formula);

        assert_eq!(cnf.clauses.len(), 1);
        assert_eq!(cnf.clauses[0].literals.len(), 2);
    }

    #[test]
    fn test_eliminate_implications() {
        // x → y becomes ¬x ∨ y
        let formula = Formula::implies(Formula::var("x"), Formula::var("y"));
        let result = eliminate_implications(&formula);

        match result {
            Formula::Or(fs) => {
                assert_eq!(fs.len(), 2);
            }
            _ => panic!("Expected Or formula"),
        }
    }

    #[test]
    fn test_dnf_from_formula() {
        // x ∧ y
        let formula = Formula::and(vec![Formula::var("x"), Formula::var("y")]);
        let dnf = Dnf::from_formula(&formula);

        assert_eq!(dnf.terms.len(), 1);
        assert_eq!(dnf.terms[0].len(), 2);
    }

    #[test]
    fn test_dnf_from_or_formula() {
        // x ∨ y
        let formula = Formula::or(vec![Formula::var("x"), Formula::var("y")]);
        let dnf = Dnf::from_formula(&formula);

        assert_eq!(dnf.terms.len(), 2);
    }

    #[test]
    fn test_cnf_roundtrip() {
        let formula = Formula::and(vec![
            Formula::var("x"),
            Formula::or(vec![Formula::var("y"), Formula::var("z")]),
        ]);

        let cnf = Cnf::from_formula(&formula);
        let back = cnf.to_formula();

        // The formulas should be logically equivalent
        // We can verify by checking a few assignments
        use std::collections::HashMap;
        let mut assignment = HashMap::new();
        assignment.insert(Variable::new("x"), true);
        assignment.insert(Variable::new("y"), false);
        assignment.insert(Variable::new("z"), true);

        assert_eq!(
            formula.evaluate(&assignment),
            back.evaluate(&assignment)
        );
    }
}
