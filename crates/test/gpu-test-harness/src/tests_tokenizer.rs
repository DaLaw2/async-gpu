//! Tokenizer validation tests — GPT-2 BPE encode/decode against known values.
//!
//! Covers: simple English, punctuation, numbers, unicode, whitespace,
//! code snippets, empty string, single character, special tokens, and more.

use gpu_host::error::Result;
use gpu_host::tokenizer::{Gpt2Tokenizer, ENDOFTEXT_TOKEN_ID, GPT2_VOCAB_SIZE};

/// Validate GPT-2 tokenizer: encode, decode, special tokens, vocab size.
pub fn run_tokenizer_test() -> Result<()> {
    println!("\n--- Tokenizer test (tokenizer.2) ---");

    let tok = Gpt2Tokenizer::new().expect("failed to create GPT-2 tokenizer");

    // 1. Vocab size
    assert_eq!(tok.vocab_size(), GPT2_VOCAB_SIZE);
    println!("  vocab_size = {} OK", tok.vocab_size());

    // 2. Encode "Hello, world!" and print token IDs
    let text = "Hello, world!";
    let ids = tok.encode(text);
    println!("  encode(\"{text}\") = {ids:?}");
    assert!(!ids.is_empty(), "encoding should produce tokens");

    // 3. Round-trip: encode then decode should recover original text
    let decoded = tok.decode(&ids).expect("decode failed");
    assert_eq!(
        decoded, text,
        "round-trip mismatch: expected \"{text}\", got \"{decoded}\""
    );
    println!("  decode round-trip OK");

    // 4. Special token <|endoftext|> should have ID 50256
    let eot_ids = tok.encode("<|endoftext|>");
    assert_eq!(
        eot_ids,
        vec![ENDOFTEXT_TOKEN_ID],
        "expected <|endoftext|> -> [50256], got {eot_ids:?}"
    );
    println!("  <|endoftext|> -> {eot_ids:?} OK");

    // 5. Longer text round-trip
    let long_text = "The quick brown fox jumps over the lazy dog. 123 + 456 = 579";
    let long_ids = tok.encode(long_text);
    let long_decoded = tok.decode(&long_ids).expect("decode failed");
    assert_eq!(long_decoded, long_text, "long text round-trip mismatch");
    println!("  long text: {} tokens, round-trip OK", long_ids.len());

    println!("--- Tokenizer test PASSED ---");
    Ok(())
}

/// Test case: input text + optional expected token IDs (None = roundtrip-only).
struct TestCase {
    label: &'static str,
    text: &'static str,
    expected_ids: Option<&'static [u32]>,
}

/// Comprehensive GPT-2 tokenizer validation (tokenizer.3).
///
/// For each of 15 test sentences:
///   1. Encode the text and verify token count is non-zero (unless empty).
///   2. If expected IDs are provided, verify exact match.
///   3. Verify decode(encode(text)) roundtrips to the original text.
pub fn run_tokenizer_validation() -> Result<()> {
    println!("\n--- Tokenizer validation (tokenizer.3) ---");

    let tok = Gpt2Tokenizer::new().expect("failed to create GPT-2 tokenizer");

    // Verify vocab size first.
    assert_eq!(tok.vocab_size(), GPT2_VOCAB_SIZE);
    println!("  vocab_size = {} OK", tok.vocab_size());

    let cases: &[TestCase] = &[
        // 1. Simple English
        TestCase {
            label: "simple English",
            text: "Hello, world!",
            expected_ids: None,
        },
        // 2. Longer English sentence
        TestCase {
            label: "longer English",
            text: "The quick brown fox jumps over the lazy dog.",
            expected_ids: None,
        },
        // 3. Punctuation-heavy
        TestCase {
            label: "punctuation",
            text: "Wait... really?! Yes! 100% sure.",
            expected_ids: None,
        },
        // 4. Numbers and arithmetic
        TestCase {
            label: "numbers",
            text: "123 + 456 = 579, and 3.14 * 2 = 6.28",
            expected_ids: None,
        },
        // 5. Unicode — CJK characters
        TestCase {
            label: "unicode CJK",
            text: "Hello \u{4f60}\u{597d}\u{4e16}\u{754c}",
            expected_ids: None,
        },
        // 6. Unicode — emoji
        TestCase {
            label: "unicode emoji",
            text: "I love Rust \u{1f980}\u{1f680}",
            expected_ids: None,
        },
        // 7. Whitespace variations
        TestCase {
            label: "whitespace",
            text: "tabs\there\nand\nnewlines  plus   spaces",
            expected_ids: None,
        },
        // 8. Code snippet
        TestCase {
            label: "code snippet",
            text: "fn main() { println!(\"hello\"); }",
            expected_ids: None,
        },
        // 9. Empty string
        TestCase {
            label: "empty string",
            text: "",
            expected_ids: Some(&[]),
        },
        // 10. Single character
        TestCase {
            label: "single char 'A'",
            text: "A",
            expected_ids: None,
        },
        // 11. Special token <|endoftext|>
        TestCase {
            label: "<|endoftext|>",
            text: "<|endoftext|>",
            expected_ids: Some(&[ENDOFTEXT_TOKEN_ID]),
        },
        // 12. Mixed special + normal text
        TestCase {
            label: "text + endoftext",
            text: "Hello<|endoftext|>world",
            expected_ids: None,
        },
        // 13. Repeated characters
        TestCase {
            label: "repeated chars",
            text: "aaaaaaaaaa",
            expected_ids: None,
        },
        // 14. JSON-like structure
        TestCase {
            label: "JSON-like",
            text: r#"{"key": "value", "num": 42}"#,
            expected_ids: None,
        },
        // 15. Long paragraph
        TestCase {
            label: "long paragraph",
            text: "Rust is a multi-paradigm, general-purpose programming language that \
                   emphasizes performance, type safety, and concurrency. It enforces memory \
                   safety without a garbage collector.",
            expected_ids: None,
        },
    ];

    let mut pass_count = 0;

    for (i, tc) in cases.iter().enumerate() {
        let ids = tok.encode(tc.text);

        // Check expected IDs if provided.
        if let Some(expected) = tc.expected_ids {
            assert_eq!(
                ids,
                expected,
                "test #{} [{}]: expected {:?}, got {:?}",
                i + 1,
                tc.label,
                expected,
                ids
            );
        }

        // Non-empty text should produce non-empty tokens.
        if !tc.text.is_empty() {
            assert!(
                !ids.is_empty(),
                "test #{} [{}]: non-empty text produced no tokens",
                i + 1,
                tc.label
            );
        }

        // Roundtrip: decode(encode(text)) == text
        let decoded = tok
            .decode(&ids)
            .unwrap_or_else(|e| panic!("test #{} [{}]: decode failed: {e}", i + 1, tc.label));
        assert_eq!(
            decoded,
            tc.text,
            "test #{} [{}]: roundtrip mismatch: expected {:?}, got {:?}",
            i + 1,
            tc.label,
            tc.text,
            decoded
        );

        println!(
            "  #{:>2} [{}]: {} tokens, roundtrip OK",
            i + 1,
            tc.label,
            ids.len()
        );
        pass_count += 1;
    }

    // Additional specific-value checks beyond the table-driven tests.
    // "Hello" is a single token in GPT-2: token 15496
    let hello_ids = tok.encode("Hello");
    assert_eq!(
        hello_ids,
        vec![15496],
        "\"Hello\" should encode to [15496], got {:?}",
        hello_ids
    );
    println!("  specific: \"Hello\" -> {:?} OK", hello_ids);

    // " the" (space + the) is token 262 in GPT-2
    let the_ids = tok.encode(" the");
    assert_eq!(
        the_ids,
        vec![262],
        "\" the\" should encode to [262], got {:?}",
        the_ids
    );
    println!("  specific: \" the\" -> {:?} OK", the_ids);

    // Single space is token 220
    let space_ids = tok.encode(" ");
    assert_eq!(
        space_ids,
        vec![220],
        "\" \" (space) should encode to [220], got {:?}",
        space_ids
    );
    println!("  specific: \" \" (space) -> {:?} OK", space_ids);

    println!(
        "--- Tokenizer validation PASSED ({} cases + 3 specific checks) ---",
        pass_count
    );
    Ok(())
}
