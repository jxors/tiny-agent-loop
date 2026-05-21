# Tiny Agent Loop
This crate provides a high-level abstraction around `async-openai` that implements a very simple agent loop.
Crates like `async-openai` provide fine-grained API access to LLMs,
but they require quite a bit of boilerplate to implement a simple agent loop.
`tiny-agent-loop` instead provides a very simple interface that you can use to implement an agent loop in a few dozen lines of code.

## When to use
Use `tiny-agent-loop` for prototypes with (local) LLMs.
If you are building something more serious, you should use one of the more feature-complete libraries instead.

## Example
```rust
use async_openai::{Client, config::OpenAIConfig};
use std::ops::ControlFlow;

use tiny_agent_loop::{Agent, FinishHandler, tool};

async fn run() {
    let client: Client<OpenAIConfig> = todo!();
    let result = Agent::new(
        "any",
        (
            tool("add", "Add two integers together", |AddArgs { a, b }| async move {
                ControlFlow::Continue(format!("Result: {}", a + b))
            }),
            tool("show_result", "Presents a result to the user", |args: ResultArgs| async move {
                ControlFlow::Break(args.result)
            }),
        ),
        FinishHandler::from(|| async {
            ControlFlow::Continue("please terminate with the show_result tool".into())
        })
    ).run(
        &client,
        "Compute the result of the following additions with the `add` tool, then show it with `show_result`:\n\nSum 5 and 7. Add another two. Also add 123871298 to the sum."
    ).await;
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
struct AddArgs {
    a: i32,
    b: i32,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
struct ResultArgs {
    result: i32,
}
```

## License
[AGPLv3](./LICENSE)