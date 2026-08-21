(() => {
  const nativeFetch = window.fetch.bind(window);
  let registerMode = false;

  window.fetch = (url, options = {}) => {
    if (url === "/tasks" && options.method === "POST" && options.body) {
      const payload = JSON.parse(options.body);
      const project = document.getElementById("projectSelect").value;
      if (project) payload.project_id = project;
      options = {...options, body: JSON.stringify(payload)};
    }
    return nativeFetch(url, options);
  };

  async function request(url, options) {
    const response = await nativeFetch(url, options);
    let body = {};
    try { body = await response.json(); } catch {}
    if (!response.ok) throw new Error(body.detail || `请求失败 (${response.status})`);
    return body;
  }

  async function loadProjects() {
    const data = await request("/projects");
    const select = document.getElementById("projectSelect");
    select.innerHTML = data.projects.map(
      project => `<option value="${project.project_id}">${project.name}</option>`
    ).join("");
    document.getElementById("projectField").classList.remove("hidden");
  }

  async function authenticated(user) {
    document.getElementById("authGate").classList.add("hidden");
    const button = document.getElementById("accountButton");
    button.textContent = `${user.display_name} · 退出`;
    button.classList.remove("hidden");
    await loadProjects();
    if (typeof refreshHistory === "function") refreshHistory();
  }

  async function bootstrap() {
    try {
      const health = await request("/health");
      if (!health.auth_enabled) return;
      try {
        const session = await request("/auth/me");
        await authenticated(session.user);
      } catch {
        document.getElementById("authGate").classList.remove("hidden");
      }
    } catch {}
  }

  document.getElementById("authMode").onclick = () => {
    registerMode = !registerMode;
    document.getElementById("nameField").classList.toggle("hidden", !registerMode);
    document.getElementById("authSubmitText").textContent = registerMode ? "注册并登录" : "登录";
    document.getElementById("authMode").textContent = registerMode ? "已有账户？返回登录" : "没有账户？立即注册";
  };

  document.getElementById("authForm").onsubmit = async event => {
    event.preventDefault();
    const error = document.getElementById("authError");
    error.textContent = "";
    const payload = {
      email: document.getElementById("authEmail").value,
      password: document.getElementById("authPassword").value,
    };
    if (registerMode) payload.display_name = document.getElementById("authName").value;
    try {
      const data = await request(registerMode ? "/auth/register" : "/auth/login", {
        method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload),
      });
      await authenticated(data.user);
    } catch (exception) { error.textContent = exception.message; }
  };

  document.getElementById("accountButton").onclick = async () => {
    await nativeFetch("/auth/logout", {method: "POST"});
    location.reload();
  };

  bootstrap();
})();
