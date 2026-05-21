use async_openai::{Client, config::OpenAIConfig};
use schemars::JsonSchema;
use serde::Deserialize;
use std::ops::ControlFlow;
use tiny_agent_loop::{Agent, FinishHandler, tool};

#[derive(Debug, Deserialize, JsonSchema)]
struct AddArgs {
    a: i32,
    b: i32,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct ResultArgs {
    result: i32,
}

#[tokio::main]
async fn main() {
    env_logger::init();

    let config = OpenAIConfig::new()
        .with_api_base("http://localhost:8080/v1")
        .with_api_key("not-needed");
    let client = Client::with_config(config);

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

    println!("\nComputed result: {}", result);
}
