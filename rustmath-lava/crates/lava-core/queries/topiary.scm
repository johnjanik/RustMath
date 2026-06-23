;; Topiary formatting queries for Magma
;;
;; ============================================================
;; Leaves: content that should not be reformatted
;; ============================================================

[
  (comment)
  (string)
  (doc_string)
] @leaf

;; ============================================================
;; Allow blank lines between top-level constructs
;; ============================================================

[
  (expression_statement)
  (assignment)
  (comment)
  (freeze)
  (forward)
  (import_directive)
  (declare_statement)
  (if_statement)
  (for_statement)
  (while_statement)
  (repeat_statement)
  (case_statement)
  (try_catch_statement)
] @allow_blank_line_before

;; ============================================================
;; Semicolons: always followed by a newline, no space before
;; ============================================================

(block
  ";" @append_hardline
)

(program
  ";" @append_hardline
)

";" @prepend_antispace

;; ============================================================
;; Function/procedure/intrinsic definitions
;; ============================================================

;; Named function definition: hardline before, space after `function`
(function_definition
  .
  "function" @append_space
  .
  name: (identifier)
) @prepend_hardline

;; Anonymous function (used as expression): no hardline, strip
;; whitespace after `function` so it abuts the parameter list.
(function_definition
  .
  "function" @append_antispace
  .
  parameters: (parameters)
)

;; Named procedure: space after `procedure`.
(procedure_definition
  .
  "procedure" @append_space
  .
  name: (identifier)
)

;; Anonymous procedure: strip whitespace after `procedure`.
(procedure_definition
  .
  "procedure" @append_antispace
  .
  parameters: (parameters)
)

(intrinsic_definition
  .
  "intrinsic" @append_space
)

;; Newline after parameters, indent body block
(function_definition
  parameters: (parameters) @append_hardline
)
(function_definition
  body: (block) @prepend_indent_start @append_indent_end
)

(function_definition
  "end" @append_space
)

(procedure_definition
  parameters: (parameters) @append_hardline
)
(procedure_definition
  body: (block) @prepend_indent_start @append_indent_end
)

(procedure_definition
  "end" @prepend_hardline @append_space
)

(intrinsic_definition
  (doc_string) @append_hardline
)
(intrinsic_definition
  body: (block) @prepend_indent_start @append_indent_end
)

(intrinsic_definition
  "end" @prepend_hardline @append_space
)

;; ============================================================
;; Control flow: if/elif/else
;; ============================================================

;; Keyword spacing — fires regardless of whether the consequence is a block.
;; A consequence of comments-only doesn't parse as (block), so combining the
;; keyword and indent rules into one pattern breaks both.
(if_statement
  "if" @append_space
  "then" @prepend_space @append_hardline
  "end" @prepend_hardline @append_space
)
(if_statement
  consequence: (block) @prepend_indent_start @append_indent_end
)

(elif_clause
  "elif" @prepend_hardline @append_space
  "then" @prepend_space @append_hardline
)
(elif_clause
  consequence: (block) @prepend_indent_start @append_indent_end
)

(else_clause
  "else" @prepend_hardline @append_hardline
)
(else_clause
  consequence: (block) @prepend_indent_start @append_indent_end
)

;; ============================================================
;; Control flow: for
;; ============================================================

(for_statement
  "for" @append_space
  "do" @prepend_space @append_hardline
  "end" @prepend_hardline @append_space
)
(for_statement
  body: (block) @prepend_indent_start @append_indent_end
)

(for_quantifier
  ":=" @prepend_space @append_space
)

(for_quantifier
  "to" @prepend_space @append_space
)

(for_quantifier
  "by" @prepend_space @append_space
)

;; range expression `a..b by c` (used inside aggregates: `[1..n by -1]`).
;; The grammar's range rule has bare 'by' (no surrounding spaces in literal),
;; so without this rule `[1..n by -1]` collapses to `[1..nby-1]`.
(range
  "by" @prepend_space @append_space
)

;; ============================================================
;; Control flow: while
;; ============================================================

(while_statement
  "while" @append_space
  "do" @prepend_space @append_hardline
  "end" @prepend_hardline @append_space
)
(while_statement
  body: (block) @prepend_indent_start @append_indent_end
)

;; ============================================================
;; Control flow: repeat/until
;; ============================================================

(repeat_statement
  "repeat" @append_hardline
  "until" @prepend_hardline @append_space
)
(repeat_statement
  body: (block) @prepend_indent_start @append_indent_end
)

;; ============================================================
;; Control flow: case
;; ============================================================

(case_statement
  "case" @append_space
)

(case_statement
  ":" @append_hardline @append_indent_start
)

(when_clause
  "when" @prepend_hardline @append_space
  ":" @append_hardline
)
(when_clause
  consequence: (block) @prepend_indent_start @append_indent_end
)

(case_statement
  "else" @prepend_hardline @append_hardline
)
(case_statement
  else: (block) @prepend_indent_start @append_indent_end
)

(case_statement
  "end" @prepend_hardline @prepend_indent_end @append_space
)

;; ============================================================
;; Control flow: try/catch
;; ============================================================

;; Opening "try" gets hardline after (anchored as first child)
(try_catch_statement
  .
  "try" @append_hardline
)

;; Indent both blocks in try/catch
(try_catch_statement
  (block) @prepend_indent_start @append_indent_end
)

(try_catch_statement
  "catch" @prepend_hardline @append_space
)

(try_catch_statement
  error: (_) @append_hardline
)

(try_catch_statement
  "end" @prepend_hardline @append_space
)



"," @prepend_antispace @append_space


;; ============================================================
;; Parentheses: no space inside
;; ============================================================

(parameters
  "(" @append_antispace
  ")" @prepend_antispace
)

(argument_list
  "(" @append_antispace
  ")" @prepend_antispace
)

(parenthesized_expression
  "(" @append_antispace
  ")" @prepend_antispace
)

;; Preserve user-chosen multi-line argument formatting.
;; @append_input_softline expands to a newline only if the comma
;; was followed by a newline in the input; otherwise it's a space
;; (which the global "," rule already provides).
;; This is the closest topiary's input-driven softline model gets
;; to "wrap if too long" — see topiary/README.md for context.
(parameters
  "," @append_input_softline
)

(argument_list
  "," @append_input_softline
)

;; ============================================================
;; Constructors — angle brackets: no space inside
;; ============================================================

(constructor
  "<" @append_antispace
  ">" @prepend_antispace
)

(recformat_constructor
  "<" @append_antispace
  ">" @prepend_antispace
)

(tuple
  "<" @append_antispace
  ">" @prepend_antispace
)

;; Constructor `|` separates options/parameters from elements:
;;   quo<F | r1, r2>  or  quo<GrpFP : F1, F2 | R : opt := val>
(constructor_elements
  "|" @prepend_space @append_space
)

;; Map constructor `:->` separates domain element from image:
;;   map< X -> Y | x :-> x^2 >
(map_constructor
  ":->" @prepend_space @append_space
)

;; two_tuple `->` (e.g., the `X -> Y` in `map<X -> Y | ...>`):
;; mirrors the `intrinsic_definition` `->` rule so the arrow has
;; consistent spacing wherever it appears as a structural separator.
(two_tuple
  "->" @prepend_space @append_space
)

;; constructor_options `:` separates the options block (e.g., the
;; `: F1, F2` in `quo<GrpFP : F1, F2 | ...>` and the trailing
;; `: opt := val`).
(constructor_options
  ":" @prepend_space @append_space
)

;; simple_assignment `name := value` appears inside constructor
;; options/elements (e.g., `R : opt := val`). Space around `:=`
;; mirrors the global assignment rule.
(simple_assignment
  ":=" @prepend_space @append_space
)

;; field_definition: "name : Type"
(field_definition
 ":" @prepend_space @append_space
 )

;; Comprehension separators ":" and "|" inside aggregates
;; (one explicit rule per aggregate node — bracket-list-of-parents matched
;; ":" globally including in require_statement, so we list each parent).
(seqenum     ":" @prepend_space @append_space)
(list        ":" @prepend_space @append_space)
(set         ":" @prepend_space @append_space)
(indexed_set ":" @prepend_space @append_space)
(formal_set  ":" @prepend_space @append_space)
(multiset    ":" @prepend_space @append_space)
(tuple       ":" @prepend_space @append_space)

(seqenum     "|" @prepend_space @append_space)
(list        "|" @prepend_space @append_space)
(set         "|" @prepend_space @append_space)
(indexed_set "|" @prepend_space @append_space)
(formal_set  "|" @prepend_space @append_space)
(multiset    "|" @prepend_space @append_space)
(tuple       "|" @prepend_space @append_space)

(iter_var
 "in" @prepend_space @append_space)

;; ============================================================
;; Aggregates — no space inside brackets
;; ============================================================

(seqenum
  "[" @append_antispace
  "]" @prepend_antispace
)

(set
  "{" @append_antispace
  "}" @prepend_antispace
)

;; ============================================================
;; Where clauses
;; ============================================================

(where_expression
 "where" @prepend_space @append_space
 operator: [":=" "is"] @prepend_space @append_space
 )

;; ============================================================
;; Statement keywords: space after
;; ============================================================

(return_statement
  "return" @append_space
)

(print_statement
  ["print" "printf" "fprintf"] @append_space
)

(vprint_statement
  ["vprint" "vprintf"] @append_space
  ":" @append_space
)

(eval_expression
  "eval" @append_space
)

;; break / continue can take an optional label identifier:
;;   break u;  break 2;  continue u;
;; The keyword needs a trailing space when followed by an identifier.
;; The existing program-level ";" @prepend_antispace rule strips the
;; space when the statement is bare (`break;`).
(break_statement
  "break" @append_space
)
(continue_statement
  "continue" @append_space
)

(error_statement
  "error" @append_space
)

;; error if cond, msg — the conditional form: `error if NOT separable, "..."`
(error_statement
  "if" @prepend_space @append_space
)

(assert_statement
  ["assert" "assert2" "assert3"] @append_space
)

(require_statement
  "require" @append_space
)

(require_statement
  ":" @append_space
)

(require_statement
  "requirege" @append_space
)

(require_statement
  "requirerange" @append_space
)

(local_statement
  "local" @append_space
)

(delete
  "delete" @append_space
)

(forward
  "forward" @append_space
)

(read_statement
  ["read" "readi"] @append_space
)

(exit_directive
  ["quit" "exit"] @append_space
)

(time_statement
  "time" @append_space
)

(vtime_statement
  "vtime" @append_space
  ":" @append_space
)

;; ============================================================
;; Declare statements
;; ============================================================

(declare_statement
  "declare" @append_space
)
(verbosity_declaration
  "verbose" @append_space
)
(attribute_declaration
  "attributes" @append_space
  ":" @prepend_antispace @append_space
)
(type_declaration
  "type" @append_space
)
(type_declaration
  ":" @prepend_antispace @append_space
)

;; ============================================================
;; Import directives
;; ============================================================

(import_directive
  "import" @append_space
)

(import_directive
  ":" @append_space
)

;; ============================================================
;; Optional parameter (name := value)
;; ============================================================

(optional_parameter
  ":=" @prepend_space @append_space
)

;; ============================================================
;; Optional argument at call sites: f(req, req : opt := val, ...)
;; ============================================================

(argument_list
  ":" @prepend_space @append_space
)

(optional_argument
  ":=" @prepend_space @append_space
)

;; ============================================================
;; Typed identifier (x :: Type)
;; ============================================================

(typed_identifier
 "::" @prepend_antispace @append_antispace
 )

;; ============================================================
;; Ternary operator (cond select x else y)
;; ============================================================

(ternary_operator
  ["select" "else"] @prepend_space @append_space
)

;; ============================================================
;; Arrow operator in intrinsic definitions
;; ============================================================

(intrinsic_definition
  "->" @prepend_space @append_space
)

;; ============================================================
;; Ref identifier (~x): no space after tilde
;; ============================================================

(ref_identifier
  "~" @append_antispace
)

;; ============================================================
;; Unary operators
;; ============================================================

(unary_operator
  operator: ["not" "assigned"] @append_space
)

(unary_operator
  operator: ["-" "+" "#" "~"] @append_antispace
)

;; ============================================================
;; Binary operators
;; ============================================================

;; High-precedence punctuation — tight, no surrounding space.
;; (Per style.md: "Space around binary operators matching order of
;; operation 3*x^2 + a".)
(binary_operator
  operator: ["*" "/" "^" "!" "!!" "@" "@@" "."] @prepend_antispace @append_antispace
)

;; Low/mid precedence and word operators — spaced.
;; ^^ (group conjugation) is precedence 7 in grammar.js, lower than
;; or/and/comparisons, so it stays spaced.
;; mod/div/cat and the rest are word operators; the lexer requires
;; whitespace so they must stay spaced.
(binary_operator
  operator: ["+" "-" "^^"
             "mod" "div" "cat"
             "join" "meet" "diff" "sdiff"
             "adj" "notadj" "subset" "notsubset"
             "in" "notin"
             "gt" "lt" "ge" "le" "eq" "ne" "cmpeq" "cmpne"] @prepend_space @append_space
)

(boolean_operator
 operator: ["or" "xor" "and"] @prepend_space @append_space)

;; ============================================================
;; Assignment
;; ============================================================

(assignment
 ":=" @prepend_space @append_space
 )

(augmented_assignment
 ["join:="  "meet:="  "diff:="  "sdiff:="  "cat:="  "*:="  "+:="  "-:="  "/:="  "^:="  "div:="  "mod:="  "and:="  "or:="  "xor:="] @prepend_space @append_space
 )

;; ============================================================
;; Comments
;; ============================================================

(comment) @prepend_input_softline
(comment) @append_hardline

;; ============================================================
;; Intrinsic definition extras
;; ============================================================

(intrinsic_definition
  (doc_string) @prepend_hardline
)
