//! HTML and MathML generation for mathematical objects

use crate::{BracketStyle, DisplayMode};

/// Format a fraction as MathML
pub fn fraction(numerator: &str, denominator: &str) -> String {
    format!(
        "<mfrac><mrow>{}</mrow><mrow>{}</mrow></mfrac>",
        numerator, denominator
    )
}

/// Format a power/exponent as MathML
pub fn power(base: &str, exponent: &str) -> String {
    format!("<msup><mrow>{}</mrow><mrow>{}</mrow></msup>", base, exponent)
}

/// Format a subscript as MathML
pub fn subscript(base: &str, sub: &str) -> String {
    format!("<msub><mrow>{}</mrow><mrow>{}</mrow></msub>", base, sub)
}

/// Format a square root as MathML
pub fn sqrt(content: &str) -> String {
    format!("<msqrt><mrow>{}</mrow></msqrt>", content)
}

/// Format an nth root as MathML
pub fn root(content: &str, n: &str) -> String {
    format!("<mroot><mrow>{}</mrow><mrow>{}</mrow></mroot>", content, n)
}

/// Format absolute value as MathML
pub fn abs(content: &str) -> String {
    format!(
        "<mrow><mo>|</mo>{}<mo>|</mo></mrow>",
        content
    )
}

/// Format a sum as MathML
pub fn sum(lower: &str, upper: &str, body: &str) -> String {
    format!(
        "<munderover><mo>&sum;</mo><mrow>{}</mrow><mrow>{}</mrow></munderover><mrow>{}</mrow>",
        lower, upper, body
    )
}

/// Format a product as MathML
pub fn product(lower: &str, upper: &str, body: &str) -> String {
    format!(
        "<munderover><mo>&prod;</mo><mrow>{}</mrow><mrow>{}</mrow></munderover><mrow>{}</mrow>",
        lower, upper, body
    )
}

/// Format an integral as MathML
pub fn integral(lower: Option<&str>, upper: Option<&str>, integrand: &str, var: &str) -> String {
    match (lower, upper) {
        (Some(l), Some(u)) => format!(
            "<msubsup><mo>&int;</mo><mrow>{}</mrow><mrow>{}</mrow></msubsup><mrow>{}</mrow><mo>d</mo><mi>{}</mi>",
            l, u, integrand, var
        ),
        _ => format!(
            "<mo>&int;</mo><mrow>{}</mrow><mo>d</mo><mi>{}</mi>",
            integrand, var
        ),
    }
}

/// Format a limit as MathML
pub fn limit(var: &str, value: &str, expr: &str) -> String {
    format!(
        "<munder><mo>lim</mo><mrow><mi>{}</mi><mo>&rarr;</mo><mrow>{}</mrow></mrow></munder><mrow>{}</mrow>",
        var, value, expr
    )
}

/// Format a function call as MathML
pub fn function(name: &str, arg: &str) -> String {
    format!(
        "<mrow><mi>{}</mi><mo>(</mo>{}<mo>)</mo></mrow>",
        name, arg
    )
}

/// Format an identifier (variable name) as MathML
pub fn identifier(name: &str) -> String {
    format!("<mi>{}</mi>", escape_html(name))
}

/// Format a number as MathML
pub fn number(value: &str) -> String {
    format!("<mn>{}</mn>", escape_html(value))
}

/// Format an operator as MathML
pub fn operator(op: &str) -> String {
    let entity = match op {
        "+" => "+",
        "-" => "&minus;",
        "*" => "&times;",
        "/" => "&divide;",
        "=" => "=",
        "<" => "&lt;",
        ">" => "&gt;",
        "<=" => "&le;",
        ">=" => "&ge;",
        "!=" => "&ne;",
        "~" => "&sim;",
        _ => op,
    };
    format!("<mo>{}</mo>", entity)
}

/// Get opening bracket for MathML
pub fn opening_bracket(style: BracketStyle) -> &'static str {
    match style {
        BracketStyle::Square => "[",
        BracketStyle::Round => "(",
        BracketStyle::Curly => "{",
        BracketStyle::Vertical => "|",
        BracketStyle::DoubleVertical => "&Vert;",
    }
}

/// Get closing bracket for MathML
pub fn closing_bracket(style: BracketStyle) -> &'static str {
    match style {
        BracketStyle::Square => "]",
        BracketStyle::Round => ")",
        BracketStyle::Curly => "}",
        BracketStyle::Vertical => "|",
        BracketStyle::DoubleVertical => "&Vert;",
    }
}

/// Format a matrix as MathML
pub fn matrix(rows: &[Vec<String>], bracket_style: BracketStyle) -> String {
    if rows.is_empty() {
        return format!(
            "<mrow><mo>{}</mo><mo>{}</mo></mrow>",
            opening_bracket(bracket_style),
            closing_bracket(bracket_style)
        );
    }

    let mut result = String::from("<mrow>");
    result.push_str(&format!("<mo>{}</mo>", opening_bracket(bracket_style)));
    result.push_str("<mtable>");

    for row in rows {
        result.push_str("<mtr>");
        for elem in row {
            result.push_str(&format!("<mtd>{}</mtd>", elem));
        }
        result.push_str("</mtr>");
    }

    result.push_str("</mtable>");
    result.push_str(&format!("<mo>{}</mo>", closing_bracket(bracket_style)));
    result.push_str("</mrow>");

    result
}

/// Format a vector as MathML
pub fn vector(elements: &[String], bracket_style: BracketStyle) -> String {
    let rows: Vec<Vec<String>> = elements.iter().map(|e| vec![e.clone()]).collect();
    matrix(&rows, bracket_style)
}

/// Wrap content in a complete MathML document
pub fn wrap_mathml(content: &str, display: DisplayMode) -> String {
    let display_attr = match display {
        DisplayMode::Display => r#" display="block""#,
        DisplayMode::Inline => "",
    };

    format!(
        r#"<math xmlns="http://www.w3.org/1998/Math/MathML"{}>{}</math>"#,
        display_attr, content
    )
}

/// Wrap content in a complete HTML document with MathML
pub fn wrap_html(content: &str, title: &str) -> String {
    format!(
        r#"<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{}</title>
    <style>
        body {{
            font-family: 'Latin Modern Math', 'STIX Two Math', 'Cambria Math', serif;
            padding: 20px;
            max-width: 800px;
            margin: 0 auto;
        }}
        math {{
            font-size: 120%;
        }}
    </style>
</head>
<body>
    {}
</body>
</html>"#,
        escape_html(title),
        content
    )
}

/// Escape HTML special characters
pub fn escape_html(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

/// Format a piecewise function as MathML
pub fn piecewise(cases: &[(String, String)]) -> String {
    let mut result = String::from("<mrow><mo>{</mo><mtable columnalign='left'>");

    for (expr, cond) in cases {
        result.push_str(&format!(
            "<mtr><mtd>{}</mtd><mtd><mtext>if </mtext>{}</mtd></mtr>",
            expr, cond
        ));
    }

    result.push_str("</mtable></mrow>");
    result
}

/// Create a space in MathML
pub fn space(width: &str) -> String {
    format!(r#"<mspace width="{}"/>"#, width)
}

/// Create a text element in MathML
pub fn text(content: &str) -> String {
    format!("<mtext>{}</mtext>", escape_html(content))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fraction() {
        let result = fraction("1", "2");
        assert!(result.contains("<mfrac>"));
        assert!(result.contains("1"));
        assert!(result.contains("2"));
    }

    #[test]
    fn test_power() {
        let result = power("x", "2");
        assert!(result.contains("<msup>"));
        assert!(result.contains("x"));
        assert!(result.contains("2"));
    }

    #[test]
    fn test_subscript() {
        let result = subscript("x", "0");
        assert!(result.contains("<msub>"));
        assert!(result.contains("x"));
        assert!(result.contains("0"));
    }

    #[test]
    fn test_sqrt() {
        let result = sqrt("2");
        assert!(result.contains("<msqrt>"));
        assert!(result.contains("2"));
    }

    #[test]
    fn test_root() {
        let result = root("8", "3");
        assert!(result.contains("<mroot>"));
        assert!(result.contains("8"));
        assert!(result.contains("3"));
    }

    #[test]
    fn test_sum() {
        let result = sum("i=1", "n", "i");
        assert!(result.contains("&sum;"));
        assert!(result.contains("i=1"));
        assert!(result.contains("n"));
    }

    #[test]
    fn test_identifier() {
        assert_eq!(identifier("x"), "<mi>x</mi>");
    }

    #[test]
    fn test_number() {
        assert_eq!(number("42"), "<mn>42</mn>");
    }

    #[test]
    fn test_operator() {
        let result = operator("+");
        assert_eq!(result, "<mo>+</mo>");

        let result = operator("*");
        assert!(result.contains("&times;"));
    }

    #[test]
    fn test_matrix() {
        let rows = vec![
            vec!["<mn>1</mn>".to_string(), "<mn>2</mn>".to_string()],
            vec!["<mn>3</mn>".to_string(), "<mn>4</mn>".to_string()],
        ];
        let result = matrix(&rows, BracketStyle::Square);
        assert!(result.contains("<mtable>"));
        assert!(result.contains("<mtr>"));
        assert!(result.contains("<mtd>"));
        assert!(result.contains("1"));
    }

    #[test]
    fn test_wrap_mathml() {
        let result = wrap_mathml("<mn>1</mn>", DisplayMode::Display);
        assert!(result.contains(r#"<math xmlns="http://www.w3.org/1998/Math/MathML""#));
        assert!(result.contains(r#"display="block""#));
        assert!(result.contains("<mn>1</mn>"));
    }

    #[test]
    fn test_escape_html() {
        assert_eq!(escape_html("<test>"), "&lt;test&gt;");
        assert_eq!(escape_html("a & b"), "a &amp; b");
    }

    #[test]
    fn test_function() {
        let result = function("sin", "<mi>x</mi>");
        assert!(result.contains("sin"));
        assert!(result.contains("x"));
        assert!(result.contains("<mo>(</mo>"));
    }
}
