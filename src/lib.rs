#![doc = include_str!("../README.md")]

use std::{iter::once, marker::PhantomData, ops::ControlFlow};

use async_openai::{
    Client,
    config::Config,
    error::OpenAIError,
    types::chat::{
        ChatCompletionMessageToolCall, ChatCompletionMessageToolCalls,
        ChatCompletionRequestAssistantMessage, ChatCompletionRequestAssistantMessageContent,
        ChatCompletionRequestAssistantMessageContentPart,
        ChatCompletionRequestDeveloperMessageContent,
        ChatCompletionRequestDeveloperMessageContentPart, ChatCompletionRequestMessage,
        ChatCompletionRequestSystemMessageContent, ChatCompletionRequestSystemMessageContentPart,
        ChatCompletionRequestToolMessage, ChatCompletionRequestToolMessageContent,
        ChatCompletionRequestToolMessageContentPart, ChatCompletionRequestUserMessage,
        ChatCompletionRequestUserMessageContent, ChatCompletionRequestUserMessageContentPart,
        ChatCompletionTool, ChatCompletionTools, CreateChatCompletionRequestArgs, FinishReason,
        FunctionObjectArgs, ResponseFormat, ResponseFormatJsonSchema,
    },
};
use futures_util::StreamExt;
use log::{debug, trace};
use schemars::{JsonSchema, generate::SchemaSettings, transform::RestrictFormats};
use serde::de::DeserializeOwned;
use serde_json::Value;

macro_rules! impl_tools {
    ($($name:ident),*) => {
        impl<R, $($name: Tool<R>),*> Tools<R> for ($($name,)*) {
            #[allow(non_snake_case)]
            fn iter<'r>(&'r mut self) -> impl Iterator<Item = &'r mut dyn Tool<R>>
                where R: 'r
            {
                let ($($name,)*) = self;

                [
                    $($name as &mut dyn Tool<R>,)*
                ].into_iter()
            }
        }
    }
}

impl_tools!(T0);
impl_tools!(T0, T1);
impl_tools!(T0, T1, T2);
impl_tools!(T0, T1, T2, T3);
impl_tools!(T0, T1, T2, T3, T4);
impl_tools!(T0, T1, T2, T3, T4, T5);
impl_tools!(T0, T1, T2, T3, T4, T5, T6);
impl_tools!(T0, T1, T2, T3, T4, T5, T6, T7);
impl_tools!(T0, T1, T2, T3, T4, T5, T6, T7, T8);
impl_tools!(T0, T1, T2, T3, T4, T5, T6, T7, T8, T9);
impl_tools!(T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10);
impl_tools!(T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11);
impl_tools!(T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12);
impl_tools!(T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13);
impl_tools!(
    T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14
);
impl_tools!(
    T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15
);

impl<
    D: JsonSchema + DeserializeOwned,
    R,
    Fut: Future<Output = ControlFlow<R, String>>,
    F: FnMut(D) -> Fut,
> Tools<R> for FnTool<D, R, Fut, F>
{
    fn iter<'r>(&'r mut self) -> impl Iterator<Item = &'r mut dyn Tool<R>>
    where
        R: 'r,
    {
        once(self as &mut dyn Tool<R>)
    }
}

pub trait Tools<R> {
    fn iter<'r>(&'r mut self) -> impl Iterator<Item = &'r mut dyn Tool<R>>
    where
        R: 'r;
}

pub trait Tool<R> {
    /// Returns the name of the tool. May not change between calls.
    fn name(&self) -> &str;

    /// The description of the tool, provided to the LLM.
    fn description(&self) -> &str;

    /// The JSON schema of the tool parameters.
    fn parameter_schema(&self) -> Value;

    /// Invokes the tool with `data` as the provided parameters.
    /// It is assumed that the LLM ensures that the parameters match the JSON schema.
    fn invoke<'lt>(
        &'lt mut self,
        data: &str,
    ) -> Box<dyn Future<Output = ControlFlow<R, String>> + 'lt>;
}

struct FnTool<D, R, Fut, F> {
    name: &'static str,
    description: &'static str,
    invoke: F,
    _phantom: PhantomData<(fn(D) -> R, Fut)>,
}

pub fn tool<
    'a,
    D: JsonSchema + DeserializeOwned,
    R,
    Fut: Future<Output = ControlFlow<R, String>>,
    F: FnMut(D) -> Fut,
>(
    name: &'static str,
    description: &'static str,
    invoke: F,
) -> impl Tool<R> {
    FnTool {
        name,
        description,
        invoke,
        _phantom: PhantomData,
    }
}

impl<
    D: JsonSchema + DeserializeOwned,
    R,
    Fut: Future<Output = ControlFlow<R, String>>,
    F: FnMut(D) -> Fut,
> Tool<R> for FnTool<D, R, Fut, F>
{
    fn name(&self) -> &str {
        self.name
    }

    fn description(&self) -> &str {
        self.description
    }

    fn parameter_schema(&self) -> Value {
        make_schema::<D>()
    }

    fn invoke<'lt>(
        &'lt mut self,
        data: &str,
    ) -> Box<dyn Future<Output = ControlFlow<R, String>> + 'lt> {
        let args = serde_json::from_str::<D>(data).unwrap();
        Box::new((self.invoke)(args)) as Box<dyn Future<Output = ControlFlow<R, String>>>
    }
}

fn make_schema<D: JsonSchema>() -> Value {
    let schema = SchemaSettings::draft2020_12()
        .with_transform(RestrictFormats::default())
        .with(|s| s.inline_subschemas = true)
        .into_generator()
        .into_root_schema_for::<D>();

    serde_json::to_value(&schema).unwrap()
}

pub struct FinishHandler<R> {
    finish: Box<dyn FnMut() -> Box<dyn Future<Output = ControlFlow<R, String>>>>,
}

impl<R> FinishHandler<R> {
    fn invoke(&mut self) -> Box<dyn Future<Output = ControlFlow<R, String>>> {
        (self.finish)()
    }
}

impl<R, Fut: Future<Output = ControlFlow<R, String>> + 'static, F: FnMut() -> Fut + 'static> From<F>
    for FinishHandler<R>
{
    fn from(mut value: F) -> Self {
        Self {
            finish: Box::new(move || Box::new((value)())),
        }
    }
}

pub struct Agent<T, R> {
    tools: T,
    listen_content: Option<Box<dyn Fn(&str)>>,
    listen_tool_call: Option<Box<dyn Fn(&str)>>,
    _phantom: PhantomData<R>,
    finish: FinishHandler<R>,
    model: String,
}

impl<R, T: Tools<R>> Agent<T, R> {
    /// Creates a new agent with the provided `tools`.
    ///
    /// `finish` is invoked when the LLM finishes without a tool call.
    /// When `ControlFlow::Break` is returned, the loop will terminate with the provided return value.
    /// When `ControlFlow::Continue(message)` is returned, the loop continues with the returned message appended to the chat history.
    pub fn new(model: impl Into<String>, tools: T, finish: FinishHandler<R>) -> Self {
        Self {
            tools,
            finish,
            listen_content: None,
            listen_tool_call: None,
            _phantom: PhantomData,
            model: model.into(),
        }
    }

    /// Adds a listener that is invoked when streamed message content is available.
    /// This is intended for debugging purposes only.
    pub fn with_content_listener(mut self, listener: Box<dyn Fn(&str)>) -> Self {
        self.listen_content = Some(listener);
        self
    }

    /// Adds a listener that is invoked when tool calling content is available.
    /// This is intended for debugging purposes only.
    /// Tools are called automatically.
    pub fn with_tool_call_listener(mut self, listener: Box<dyn Fn(&str)>) -> Self {
        self.listen_tool_call = Some(listener);
        self
    }

    /// Convenience function that calls [`Self::start_session`] followed by [`Session::run`].
    /// 
    /// See [`Session::run`].
    pub async fn run<C: Config>(&mut self, client: &Client<C>, prompt: &str) -> R {
        self.start_session(prompt).run(client).await
    }

    /// Start a new chat session with the provided prompt.
    ///
    /// Keeps a reference to the agent rather than consuming it, such that the agent can be re-used later.
    pub fn start_session<'agent>(
        &'agent mut self,
        prompt: &str,
    ) -> Session<&'agent mut Agent<T, R>, T, R> {
        Session {
            agent: self,
            messages: vec![ChatCompletionRequestUserMessage::from(prompt).into()],
            _phantom: PhantomData,
        }
    }

    /// Start a new chat session with the provided prompt.
    ///
    /// Consumes the [`Agent`], creating a session without lifetimes.
    pub fn into_session(self, prompt: &str) -> Session<Agent<T, R>, T, R> {
        Session {
            agent: self,
            messages: vec![ChatCompletionRequestUserMessage::from(prompt).into()],
            _phantom: PhantomData,
        }
    }
}

impl<T, R> AsMut<Agent<T, R>> for Agent<T, R> {
    fn as_mut(&mut self) -> &mut Agent<T, R> {
        self
    }
}

pub struct Session<A, T, R> {
    agent: A,
    messages: Vec<ChatCompletionRequestMessage>,
    _phantom: PhantomData<(R, T)>,
}

impl<R, T: Tools<R>, A: AsMut<Agent<T, R>>> Session<A, T, R> {
    /// Append an additional message to the chat.
    pub fn add_message(
        &mut self,
        message: impl Into<ChatCompletionRequestUserMessage>,
    ) -> &mut Self {
        self.messages.push(message.into().into());
        self
    }

    /// Runs the agent loop. If tools were provided, they may be called.
    ///
    /// Terminates when a tool returns ControlFlow::Break.
    ///
    /// It is assumed that any single message or tool response always fits in the context window fully.
    /// If this is not the case, this function may crash.
    pub async fn run<C: Config>(&mut self, client: &Client<C>) -> R {
        loop {
            let tools = self
                .agent
                .as_mut()
                .tools
                .iter()
                .map(|tool| {
                    ChatCompletionTools::Function(ChatCompletionTool {
                        function: FunctionObjectArgs::default()
                            .name(tool.name())
                            .description(tool.description())
                            .parameters(tool.parameter_schema())
                            .build()
                            .unwrap(),
                    })
                })
                .collect::<Vec<_>>();

            let request = CreateChatCompletionRequestArgs::default()
                .messages(self.messages.clone())
                .tools(tools)
                .model(&self.agent.as_mut().model)
                .build()
                .unwrap();

            let mut stream = client.chat().create_stream(request).await.unwrap();
            let mut tool_calls = Vec::new();
            let mut execution_results = Vec::new();

            while let Some(result) = stream.next().await {
                let response = match result {
                    Ok(response) => response,
                    Err(OpenAIError::JSONDeserialize(e, json)) => {
                        let json: Value = serde_json::from_str(&json).unwrap();
                        if let Some(err) = json.get("error")
                            && let Some(ty) = err.get("type")
                            && let Some(ty) = ty.as_str()
                            && ty == "exceed_context_size_error"
                        {
                            // When we run out of context, we summarize older messages.
                            // It is assumed that a single message will always fit the context size.
                            let num_to_summarize = {
                                // messages: Vec<async_openai::types::chat::ChatCompletionRequestMessage>
                                let total_characters =
                                    self.messages.iter().map(get_message_len).sum::<usize>();
                                let min_to_summarize = total_characters / 2;
                                let mut running_sum = 0;
                                self.messages
                                    .iter()
                                    .enumerate()
                                    .take_while(|(n, m)| {
                                        // 1. Ensure we leave a buffer of recent messages (at least 3)
                                        // 2. Ensure we don't exceed the target summarization length

                                        let limit = self.messages.len().saturating_sub(3).max(1);
                                        let content_len = get_message_len(m);

                                        // We continue taking messages as long as:
                                        // We haven't reached the index limit AND we haven't reached the char limit
                                        if *n < limit && running_sum < min_to_summarize {
                                            running_sum += content_len;
                                            true
                                        } else {
                                            false
                                        }
                                    })
                                    .count()
                            };
                            let mut messages_to_summarize =
                                self.messages.drain(..num_to_summarize).collect::<Vec<_>>();
                            debug!(
                                "Context size exceeded ({err:?}), attempting summary of messages: {messages_to_summarize:#?}."
                            );

                            messages_to_summarize.push(
                                ChatCompletionRequestUserMessage::from(include_str!("summary.txt"))
                                    .into(),
                            );

                            // TODO: This will panic if there is just a single message left that doesn't fit the context window.
                            let request = CreateChatCompletionRequestArgs::default()
                                .messages(messages_to_summarize)
                                .model(&self.agent.as_mut().model)
                                .build()
                                .unwrap();
                            let response = client.chat().create(request).await.unwrap();
                            trace!("Summarized to: {response:#?}");
                            for (index, choice) in response.choices.into_iter().enumerate() {
                                self.messages.insert(
                                    index,
                                    ChatCompletionRequestUserMessage {
                                        content: ChatCompletionRequestUserMessageContent::Text(
                                            choice.message.content.unwrap(),
                                        ),
                                        ..Default::default()
                                    }
                                    .into(),
                                );
                            }

                            break;
                        } else {
                            panic!("Unexpected error: failed to deserialize JSON: {e} in {json}");
                        }
                    }
                    Err(e) => panic!("Unexpected error: {e}"),
                };

                for choice in response.choices {
                    if let Some(content) = &choice.delta.content {
                        trace!("Agent: {content:?}");
                        if let Some(f) = &self.agent.as_mut().listen_content {
                            (f)(&content);
                        }
                    }

                    if let Some(tool_call_chunks) = choice.delta.tool_calls {
                        for chunk in tool_call_chunks {
                            let index = chunk.index as usize;
                            if tool_calls.len() <= index {
                                tool_calls.resize(
                                    index + 1,
                                    ChatCompletionMessageToolCall {
                                        id: String::new(),
                                        function: Default::default(),
                                    },
                                );
                            }

                            let tool_call = &mut tool_calls[index];
                            if let Some(id) = chunk.id {
                                tool_call.id = id;
                            }

                            if let Some(function_chunk) = chunk.function {
                                if let Some(name) = function_chunk.name {
                                    if let Some(f) = &self.agent.as_mut().listen_tool_call {
                                        (f)(&name);
                                    }

                                    tool_call.function.name = name;
                                }
                                if let Some(arguments) = function_chunk.arguments {
                                    if let Some(f) = &self.agent.as_mut().listen_tool_call {
                                        (f)(&arguments);
                                    }

                                    tool_call.function.arguments.push_str(&arguments);
                                }
                            }
                        }
                    }

                    match choice.finish_reason {
                        Some(FinishReason::ToolCalls) => {
                            for tool_call in tool_calls.iter() {
                                let name = tool_call.function.name.clone();
                                let tool_call_id = tool_call.id.clone();
                                let result = {
                                    debug!("Executing tool: {name}");
                                    let tool = self
                                        .agent
                                        .as_mut()
                                        .tools
                                        .iter()
                                        .find(|tool| tool.name() == name);

                                    match tool {
                                        Some(tool) => {
                                            let future = tool.invoke(&tool_call.function.arguments);
                                            let pinned = Box::into_pin(future);
                                            pinned.await
                                        }
                                        None => ControlFlow::Continue(String::from(
                                            "ERROR: Tool not found",
                                        )),
                                    }
                                };

                                match result {
                                    ControlFlow::Continue(result) => {
                                        execution_results.push((tool_call_id, result))
                                    }
                                    ControlFlow::Break(val) => return val,
                                }
                            }
                        }
                        Some(FinishReason::Stop | FinishReason::Length) => {
                            let future = self.agent.as_mut().finish.invoke();
                            let pinned = Box::into_pin(future);
                            let result = pinned.await;

                            match result {
                                ControlFlow::Continue(result) => {
                                    self.messages.push(
                                        ChatCompletionRequestUserMessage::from(result).into(),
                                    );
                                }
                                ControlFlow::Break(val) => return val,
                            }
                        }
                        Some(reason) => panic!("Unexpected stop reason: {reason:?}"),
                        _ => (),
                    }
                }
            }

            if !execution_results.is_empty() {
                let assistant_tool_calls: Vec<ChatCompletionMessageToolCalls> =
                    tool_calls.iter().map(|tc| tc.clone().into()).collect();
                self.messages.push(
                    ChatCompletionRequestAssistantMessage {
                        content: None,
                        tool_calls: Some(assistant_tool_calls),
                        ..Default::default()
                    }
                    .into(),
                );

                for (tool_call_id, response) in execution_results {
                    self.messages.push(
                        ChatCompletionRequestToolMessage {
                            content: response.to_string().into(),
                            tool_call_id,
                        }
                        .into(),
                    );
                }
            }
        }
    }
}

fn get_message_len(m: &ChatCompletionRequestMessage) -> usize {
    match m {
        ChatCompletionRequestMessage::System(m) => match &m.content {
            ChatCompletionRequestSystemMessageContent::Text(t) => t.len(),
            ChatCompletionRequestSystemMessageContent::Array(a) => a
                .iter()
                .map(|m| match m {
                    ChatCompletionRequestSystemMessageContentPart::Text(t) => t.text.len(),
                })
                .sum::<usize>(),
        },
        ChatCompletionRequestMessage::User(m) => match &m.content {
            ChatCompletionRequestUserMessageContent::Text(t) => t.len(),
            ChatCompletionRequestUserMessageContent::Array(parts) => parts
                .iter()
                .map(|p| match p {
                    ChatCompletionRequestUserMessageContentPart::Text(t) => t.text.len(),
                    _ => 0,
                })
                .sum(),
        },
        ChatCompletionRequestMessage::Assistant(m) => m
            .content
            .as_ref()
            .map(|c| match c {
                ChatCompletionRequestAssistantMessageContent::Text(t) => t.len(),
                ChatCompletionRequestAssistantMessageContent::Array(a) => a
                    .iter()
                    .map(|m| match m {
                        ChatCompletionRequestAssistantMessageContentPart::Text(t) => t.text.len(),
                        ChatCompletionRequestAssistantMessageContentPart::Refusal(_) => 0,
                    })
                    .sum::<usize>(),
            })
            .unwrap_or(0),
        ChatCompletionRequestMessage::Tool(m) => match &m.content {
            ChatCompletionRequestToolMessageContent::Text(t) => t.len(),
            ChatCompletionRequestToolMessageContent::Array(a) => a
                .iter()
                .map(|m| match m {
                    ChatCompletionRequestToolMessageContentPart::Text(t) => t.text.len(),
                })
                .sum::<usize>(),
        },
        ChatCompletionRequestMessage::Function(m) => {
            m.content.as_ref().map(|c| c.len()).unwrap_or(0)
        }
        ChatCompletionRequestMessage::Developer(m) => match &m.content {
            ChatCompletionRequestDeveloperMessageContent::Text(t) => t.len(),
            ChatCompletionRequestDeveloperMessageContent::Array(a) => a
                .iter()
                .map(|m| match m {
                    ChatCompletionRequestDeveloperMessageContentPart::Text(t) => t.text.len(),
                })
                .sum::<usize>(),
        },
    }
}

/// Invokes the LLM once, generating an output object directly.
/// No tool calls are supported.
pub async fn one_shot<C: Config, R: JsonSchema + DeserializeOwned>(
    client: &Client<C>,
    model: &str,
    prompt: &str,
) -> R {
    let request = CreateChatCompletionRequestArgs::default()
        .messages(vec![ChatCompletionRequestUserMessage::from(prompt).into()])
        .model(model)
        .response_format(ResponseFormat::JsonSchema {
            json_schema: ResponseFormatJsonSchema {
                name: format!("Result"),
                description: None,
                schema: Some(make_schema::<R>()),
                strict: Some(true),
            },
        })
        .build()
        .unwrap();
    let response = client.chat().create(request).await.unwrap();
    let text = response
        .choices
        .into_iter()
        .map(|choice| choice.message.content.unwrap())
        .collect::<String>();

    serde_json::from_str(&text).unwrap()
}
