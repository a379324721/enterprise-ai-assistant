import React, {useCallback, useEffect, useMemo, useRef, useState} from "react";
import {createRoot} from "react-dom/client";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import "./styles.css";
import "./streaming.css";

type Task = {id: string; title: string; domain: string; objective: string; status: string};
type Confirmation = {confirmation_id: string; summary: string; action: string; payload: Record<string, unknown>};
type Result = {
  conversation_id: string; status: string; answer: string; user_goal: string;
  tasks: Task[]; artifacts: Record<string, unknown>; pending_confirmation?: Confirmation | null;
};
type Message = {role: "user" | "assistant"; text: string; index?: number};
type SseMessage = {event: string; data: unknown};
type Session = {token: string; userId: string; displayName: string; conversationId: string};

const examples = ["下周三到周五去上海出差帮我申请，顺便订个上海分部的会议室周四上午开会", "我还有多少年假？下周五请一天年假", "会议室最长能订几个小时"];
const SESSION_KEY = "eaa.session";
const PAGE_SIZE = 20;

function readSession(): Session | null {
  try {
    const raw = window.localStorage.getItem(SESSION_KEY);
    return raw ? JSON.parse(raw) as Session : null;
  } catch { return null; }
}

function MarkdownMessage({text}: {text: string}) {
  return <ReactMarkdown
    remarkPlugins={[remarkGfm]}
    components={{
      a: ({href, children}) => <a href={href} target="_blank" rel="noreferrer noopener">{children}</a>,
    }}
  >{text}</ReactMarkdown>;
}

/**
 * 取服务端的 detail 作为提示。
 *
 * FastAPI 的校验失败（422）返回的 detail 是一个数组而不是字符串，直接当字符串用
 * 会在界面上渲染成 [object Object]，所以这里按两种形状分别取。
 */
async function readError(response: Response, fallback: string): Promise<string> {
  const body = await response.text();
  let detail: unknown;
  try {
    detail = (JSON.parse(body) as {detail?: unknown}).detail;
  } catch { return body.trim() || fallback; }
  if (typeof detail === "string" && detail) return detail;
  if (Array.isArray(detail)) {
    const messages = detail
      .map((item) => (item as {msg?: string}).msg)
      .filter((item): item is string => Boolean(item));
    if (messages.length) return messages.join("；");
  }
  return fallback;
}

/** 把异常翻成用户看得懂的一句话。fetch 连不上服务端时抛 TypeError。 */
function describeFailure(issue: unknown, fallback = "加载失败"): string {
  if (issue instanceof TypeError) return "无法连接服务器，请确认后端已启动";
  return issue instanceof Error && issue.message ? issue.message : fallback;
}

async function consumeSse(
  response: Response,
  onEvent: (message: SseMessage) => void,
): Promise<void> {
  if (!response.ok) throw new Error(await readError(response, `流式请求失败（HTTP ${response.status}）`));
  if (!response.body) throw new Error("浏览器没有收到可读取的响应流");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const {done, value} = await reader.read();
    buffer += decoder.decode(value, {stream: !done});
    const blocks = buffer.split(/\r?\n\r?\n/);
    buffer = blocks.pop() ?? "";
    for (const block of blocks) {
      let event = "message";
      const dataLines: string[] = [];
      for (const line of block.split(/\r?\n/)) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
      }
      if (!dataLines.length) continue;
      onEvent({event, data: JSON.parse(dataLines.join("\n")) as unknown});
    }
    if (done) break;
  }
}

/** 演示登录：只有名字，没有凭据。正式部署由企业 SSO 取代这一屏。 */
function LoginView({onSignedIn}: {onSignedIn: (session: Session) => void}) {
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function submit() {
    const trimmed = name.trim();
    if (!trimmed || busy) return;
    setBusy(true); setError("");
    try {
      const response = await fetch("/api/v1/auth/login", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({name: trimmed}),
      });
      if (!response.ok) {
        // 名字不存在会被直接创建，所以 404 只剩一种含义：接口没开。
        throw new Error(await readError(
          response,
          response.status === 404
            ? "演示登录未开启，请确认服务端的 DEMO_LOGIN_ENABLED"
            : `进入失败（HTTP ${response.status}）`,
        ));
      }
      const body = await response.json() as {
        access_token: string; user_id: string; display_name: string; conversation_id: string;
      };
      const session: Session = {
        token: body.access_token,
        userId: body.user_id,
        displayName: body.display_name,
        conversationId: body.conversation_id,
      };
      window.localStorage.setItem(SESSION_KEY, JSON.stringify(session));
      onSignedIn(session);
    } catch (issue) {
      setError(describeFailure(issue, "进入失败"));
    } finally { setBusy(false); }
  }

  return <main className="loginPage">
    <div className="loginCard">
      <div className="brandMark">E</div>
      <h1>Enterprise AI Assistant</h1>
      <p>输入你的名字即可开始。演示环境不设密码，一个名字对应一个用户，下次用同一个名字会回到同一个会话。</p>
      <form onSubmit={(event) => { event.preventDefault(); void submit(); }}>
        <input value={name} onChange={(event) => setName(event.target.value)} placeholder="你的名字" autoFocus maxLength={64}/>
        <button className="approve" disabled={busy || !name.trim()}>{busy ? "正在进入…" : "进入"}</button>
      </form>
      {error && <p className="loginError">{error}</p>}
    </div>
  </main>;
}

function ChatView({session, onSignOut}: {session: Session; onSignOut: () => void}) {
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<Result | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [progress, setProgress] = useState("");
  const [hasMore, setHasMore] = useState(false);
  const [loadingHistory, setLoadingHistory] = useState(false);
  const [loadError, setLoadError] = useState("");
  const activeAnswerId = useRef<string | null>(null);

  const authHeaders = useMemo(
    () => ({"Content-Type": "application/json", Authorization: `Bearer ${session.token}`}),
    [session.token],
  );

  // 令牌过期后留在界面上只会不断报错，直接退回登录页重新取。
  const guard = useCallback((response: Response) => {
    if (response.status === 401) { onSignOut(); throw new Error("登录已过期，请重新登录"); }
    return response;
  }, [onSignOut]);

  /** 取一页历史，返回本页条数，供调用方判断会话是否已经有内容。 */
  const loadHistory = useCallback(async (before?: number, signal?: AbortSignal) => {
    const query = new URLSearchParams({limit: String(PAGE_SIZE)});
    if (before !== undefined) query.set("before", String(before));
    const response = guard(await fetch(
      `/api/v1/conversations/${session.conversationId}/messages?${query}`,
      {headers: authHeaders, signal},
    ));
    if (!response.ok) return 0;
    const body = await response.json() as {messages: Required<Message>[]; has_more: boolean};
    setHasMore(body.has_more);
    setMessages((old) => before === undefined ? body.messages : [...body.messages, ...old]);
    return body.messages.length;
  }, [authHeaders, guard, session.conversationId]);

  // 进入会话时补齐最近一页历史，并把上一轮未完成的确认重新摆出来——
  // 演示时刷新页面不该让一个待确认的写操作凭空消失。
  const enterConversation = useCallback(async (signal?: AbortSignal) => {
    setLoadingHistory(true); setLoadError("");
    try {
      const loaded = await loadHistory(undefined, signal);
      // 会话快照只用来恢复待确认操作和任务面板，历史为空时二者必然都不存在。
      // 新用户的会话尚未落检查点，这一请求只会换回一个 404。
      if (loaded > 0) {
        const snapshot = await fetch(
          `/api/v1/conversations/${session.conversationId}`,
          {headers: authHeaders, signal},
        );
        if (snapshot.ok) setResult(await snapshot.json() as Result);
      }
    } catch (issue) {
      // 中止来自 effect 清理，不是故障；其余情况必须说出来，否则后端没起时
      // 界面只是一片空白，看起来像历史被清掉了。
      if (signal?.aborted) return;
      setLoadError(describeFailure(issue, "历史消息加载失败"));
    } finally {
      if (!signal?.aborted) setLoadingHistory(false);
    }
  }, [authHeaders, loadHistory, session.conversationId]);

  useEffect(() => {
    // StrictMode 在开发模式下会把 effect 跑两遍，切换用户也会重跑；没有中止信号的话，
    // 先发出的那次响应可能后到并覆盖新一次的结果。
    const controller = new AbortController();
    void enterConversation(controller.signal);
    return () => controller.abort();
  }, [enterConversation]);

  async function loadEarlier() {
    const earliest = messages.find((message) => message.index !== undefined)?.index;
    if (earliest === undefined || loadingHistory) return;
    setLoadingHistory(true);
    try { await loadHistory(earliest); } finally { setLoadingHistory(false); }
  }

  function handleStreamEvent({event, data}: SseMessage) {
    if (event === "progress") {
      setProgress((data as {message: string}).message);
    } else if (event === "answer_start") {
      const messageId = (data as {message_id: string}).message_id;
      if (activeAnswerId.current && activeAnswerId.current !== messageId) {
        setMessages((old) => old.map((message, index) => index === old.length - 1 && message.text ? {...message, text: `${message.text}\n\n`} : message));
      }
      activeAnswerId.current = messageId;
    } else if (event === "token") {
      const chunk = (data as {content: string}).content;
      setMessages((old) => old.map((message, index) => index === old.length - 1 ? {...message, text: message.text + chunk} : message));
    } else if (event === "done") {
      const completed = data as Result;
      setResult(completed);
      setMessages((old) => {
        const last = old.at(-1);
        if (!last || last.role !== "assistant" || last.text) return old;
        if (completed.answer) {
          return old.map((message, index) => index === old.length - 1 ? {...message, text: completed.answer} : message);
        }
        if (completed.status === "waiting_confirmation") return old.slice(0, -1);
        return old.map((message, index) => index === old.length - 1 ? {...message, text: "未生成有效回复，请重试。"} : message);
      });
    } else if (event === "error") {
      throw new Error((data as {message: string}).message);
    }
  }

  async function send() {
    if (!input.trim() || busy) return;
    const text = input.trim(); setInput(""); setBusy(true); setProgress("正在连接智能助手");
    setMessages((old) => [...old, {role: "user", text}, {role: "assistant", text: ""}]);
    try {
      const response = guard(await fetch("/api/v1/chat/stream", {
        method: "POST", headers: authHeaders,
        body: JSON.stringify({message: text, request_id: crypto.randomUUID(), conversation_id: session.conversationId}),
      }));
      await consumeSse(response, handleStreamEvent);
    } catch (issue) {
      const message = describeFailure(issue, "系统异常");
      setMessages((old) => old.map((item, index) => index === old.length - 1 ? {...item, text: item.text || message} : item));
    } finally { setBusy(false); setProgress(""); activeAnswerId.current = null; }
  }

  async function confirm(approved: boolean) {
    if (!result?.pending_confirmation || busy) return;
    setBusy(true); setProgress(approved ? "正在确认并恢复任务" : "正在取消操作");
    setMessages((old) => [...old, {role: "assistant", text: ""}]);
    try {
      const response = guard(await fetch(`/api/v1/conversations/${session.conversationId}/confirm/stream`, {
        method: "POST", headers: authHeaders,
        body: JSON.stringify({confirmation_id: result.pending_confirmation.confirmation_id, approved}),
      }));
      await consumeSse(response, handleStreamEvent);
    } catch (issue) {
      const message = describeFailure(issue, "系统异常");
      setMessages((old) => old.map((item, index) => index === old.length - 1 ? {...item, text: item.text || message} : item));
    } finally { setBusy(false); setProgress(""); activeAnswerId.current = null; }
  }

  return <main>
    <header><div className="brandMark">E</div><div><h1>Enterprise AI Assistant</h1><p>企业事务，一个对话完成</p></div>
      <span className="online">● {session.displayName}</span>
      <button className="signOut" onClick={onSignOut}>退出</button>
    </header>
    <section className="layout">
      <div className="chatPanel">
        <div className="intro"><span>AI</span><div><strong>你好{session.displayName ? `，${session.displayName}` : ""}，我是企业智能助手</strong><p>我可以协助差旅、报销、请假和制度查询。涉及提交的操作会先请你确认。</p></div></div>
        {messages.length === 0 && !loadingHistory && <div className="examples">{examples.map((item) => <button key={item} onClick={() => setInput(item)}>{item}<b>↗</b></button>)}</div>}
        {loadError && <div className="loadBanner"><span>{loadError}</span><button disabled={loadingHistory} onClick={() => void enterConversation()}>重试</button></div>}
        <div className="messages">
          {hasMore && <button className="loadEarlier" disabled={loadingHistory} onClick={() => void loadEarlier()}>{loadingHistory ? "加载中…" : "加载更早的消息"}</button>}
          {messages.map((message, index) => <div key={message.index ?? `live-${index}`} className={`message ${message.role}`}>{message.role === "assistant" ? <MarkdownMessage text={message.text}/> : message.text}{busy && index === messages.length - 1 && message.role === "assistant" && <span className="cursor"/>}</div>)}
          {busy && <div className="thinking">{progress || "正在处理…"}</div>}
        </div>
        {result?.pending_confirmation && <div className="confirmCard"><div className="risk">需要你的确认</div><strong>{result.pending_confirmation.summary}</strong><p>系统只会在你确认后执行该操作。</p><div><button className="cancel" disabled={busy} onClick={() => void confirm(false)}>取消</button><button className="approve" disabled={busy} onClick={() => void confirm(true)}>确认执行</button></div></div>}
        <form onSubmit={(event) => { event.preventDefault(); void send(); }}>
          <textarea
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={(event) => {
              // 输入法组合期间的回车是在确认候选词，不能当成发送——中文拼音下
              // 每选一次词都会把半截话发出去。
              if (event.key !== "Enter" || event.shiftKey || event.nativeEvent.isComposing) return;
              event.preventDefault();
              void send();
            }}
            placeholder="描述你想办理的事情…（回车发送，Shift + 回车换行）"
            rows={2}
          />
          <button disabled={busy}>发送</button>
        </form>
      </div>
      <aside><div className="asideHead"><span>任务执行</span><small>{result ? `${result.tasks.filter(t => t.status === "completed").length}/${result.tasks.length}` : "0/0"}</small></div>
        {!result && <div className="empty"><i>⌁</i><p>发送请求后，这里会展示 AI 拆解出的任务及执行进度。</p></div>}
        {result && <><div className="goal"><small>理解到的目标</small><p>{result.user_goal}</p></div><div className="taskList">{result.tasks.map((task, index) => <div className="task" key={task.id}><span className={task.status}>{task.status === "completed" ? "✓" : index + 1}</span><div><strong>{task.title}</strong><small>{task.domain}</small></div><em>{({completed:"已完成",running:"执行中",waiting_confirmation:"待确认",waiting_input:"待补充",pending:"等待中",rejected:"已取消",failed:"失败"} as Record<string,string>)[task.status] || task.status}</em></div>)}</div></>}
      </aside>
    </section>
  </main>;
}

function App() {
  const [session, setSession] = useState<Session | null>(readSession);

  function signOut() {
    window.localStorage.removeItem(SESSION_KEY);
    setSession(null);
  }

  if (!session) return <LoginView onSignedIn={setSession}/>;
  // 换人登录要重建整棵子树，否则上一位的消息和任务面板会残留。
  return <ChatView key={session.userId} session={session} onSignOut={signOut}/>;
}

createRoot(document.getElementById("root")!).render(<React.StrictMode><App/></React.StrictMode>);
