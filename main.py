import webview
import threading
import os
import sys
from flask import Flask, request, jsonify
from flask_cors import CORS
from llama_cpp import Llama
import re
from threading import Lock
import time
import socket

# Get base path for PyInstaller
def get_base_path():
    if getattr(sys, 'frozen', False):
        return sys._MEIPASS
    else:
        return os.path.dirname(os.path.abspath(__file__))

BASE_PATH = get_base_path()

# Flask app
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# Global variables
llm = None
model_loaded = False
model_lock = Lock()
flask_started = False

SYSTEM_PROMPT = """You are R3KON GPT, a professional cybersecurity assistant.
CRITICAL RULES:
1. ALWAYS respond in English only. Never use Chinese or any other language.
2. Stay strictly on topic related to cybersecurity, programming, or the user's question.
3. Keep responses clear, concise, and professional.
4. Use structured formatting: bullet points, numbered lists, or paragraphs as appropriate.
5. Never repeat yourself or generate repetitive content.
6. If asked something off-topic, politely redirect to cybersecurity topics.
"""

def find_free_port():
    """Find a free port to use"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

def load_model():
    """Load the AI model"""
    global llm, model_loaded
    
    try:
        # Find model in multiple locations
        model_filename = "qwen1.5-1.8b-chat-q4_k_m.gguf"
        possible_paths = [
            os.path.join(BASE_PATH, "model", model_filename),
            os.path.join(BASE_PATH, model_filename),
            os.path.join(os.path.dirname(sys.executable), "model", model_filename),
            os.path.join(os.getcwd(), "model", model_filename),
        ]
        
        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if not model_path:
            print(f"ERROR: Model not found at any location")
            print(f"Searched: {possible_paths}")
            return False
        
        print(f"Loading model from: {model_path}")
        
        llm = Llama(
            model_path=model_path,
            n_ctx=3072,
            n_threads=8,
            n_batch=512,
            verbose=False,
            use_mlock=True,
            use_mmap=True,
        )
        
        model_loaded = True
        print("Model loaded successfully!")
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return False

def generate_response(prompt, config, history):
    """Generate response from model"""
    if not model_loaded:
        return {"error": "Model not loaded"}
    
    # Build context
    context_parts = [SYSTEM_PROMPT]
    
    if config.get('sessionMemory') and history:
        context_parts.append("\n--- Recent Conversation ---")
        for turn in history[-5:]:
            context_parts.append(f"User: {turn['user']}")
            context_parts.append(f"Assistant: {turn['assistant']}")
    
    context_parts.append(f"\nUser: {prompt}")
    context_parts.append("Assistant:")
    
    full_prompt = '\n'.join(context_parts)
    
    # Token limits
    token_limits = {"short": 300, "medium": 600, "long": 1000}
    max_tokens = token_limits.get(config.get('responseLength', 'medium'), 600)
    
    try:
        with model_lock:
            response = llm(
                full_prompt,
                max_tokens=max_tokens,
                stop=["User:", "\n\nUser:", "Assistant:"],
                echo=False,
                temperature=0.7,
                top_p=0.9,
                top_k=40,
                repeat_penalty=1.2,
                frequency_penalty=0.3,
                presence_penalty=0.3,
            )
        
        bot_reply = response["choices"][0]["text"].strip()
        
        # Filter Chinese
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', bot_reply))
        total_chars = len(bot_reply.replace(' ', '').replace('\n', ''))
        
        if total_chars > 0 and (chinese_chars / total_chars) > 0.3:
            bot_reply = "I apologize, but I can only respond in English."
        else:
            bot_reply = re.sub(r'[\u4e00-\u9fff]+', '', bot_reply)
            bot_reply = re.sub(r'\n\n+', '\n\n', bot_reply).strip()
        
        # Remove repetition
        lines = bot_reply.split('\n')
        unique_lines = []
        for line in lines:
            if line.strip() and (not unique_lines or line not in unique_lines[-2:]):
                unique_lines.append(line)
        
        bot_reply = '\n'.join(unique_lines)
        
        if len(bot_reply) < 10:
            bot_reply = "I encountered an issue. Please try rephrasing your question."
        
        return {"response": bot_reply}
        
    except Exception as e:
        print(f"Error generating response: {e}")
        return {"error": str(e)}

@app.route('/')
def index():
    """Serve the HTML page"""
    # Embed HTML directly to avoid path issues
    html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>R3KON GPT - Cybersecurity Assistant</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        :root {
            --primary-bg: #0F0F0F;
            --secondary-bg: #1A1A1A;
            --sidebar-bg: #1E1E1E;
            --input-bg: #2A2A2A;
            --text-primary: #FFFFFF;
            --text-secondary: #B0B0B0;
            --accent-blue: #00D4FF;
            --accent-green: #00FF9D;
            --accent-yellow: #FFD700;
            --accent-red: #FF4466;
            --border-color: #333333;
            --button-bg: #0078D4;
            --button-hover: #005A9E;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: var(--primary-bg);
            color: var(--text-primary);
            height: 100vh;
            overflow: hidden;
        }

        body.light-theme {
            --primary-bg: #FFFFFF;
            --secondary-bg: #F0F0F0;
            --sidebar-bg: #E5E5E5;
            --input-bg: #F5F5F5;
            --text-primary: #000000;
            --text-secondary: #666666;
            --border-color: #CCCCCC;
            --button-bg: #0078D4;
            --button-hover: #005A9E;
        }

        .container {
            display: flex;
            height: 100vh;
        }

        .sidebar {
            width: 250px;
            background: var(--sidebar-bg);
            border-right: 1px solid var(--border-color);
            display: flex;
            flex-direction: column;
            padding: 20px;
            overflow-y: auto;
        }

        .sidebar-title {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 20px;
            text-align: center;
        }

        .settings-section {
            margin-bottom: 25px;
        }

        .settings-label {
            font-size: 14px;
            color: var(--text-secondary);
            margin-bottom: 8px;
            display: block;
        }

        .theme-buttons, .font-buttons {
            display: flex;
            gap: 8px;
        }

        .btn {
            padding: 8px 12px;
            border: 1px solid var(--text-primary);
            border-radius: 4px;
            background: transparent;
            color: var(--text-primary);
            cursor: pointer;
            font-size: 13px;
            font-weight: 500;
            transition: all 0.3s;
        }

        .btn:hover {
            background: var(--text-primary);
            color: var(--primary-bg);
            transform: translateY(-1px);
        }

        .btn-full {
            width: 100%;
            margin-top: 5px;
        }

        select, input[type="checkbox"] {
            width: 100%;
            padding: 8px;
            background: var(--input-bg);
            color: var(--text-primary);
            border: 1px solid var(--border-color);
            border-radius: 4px;
            font-size: 13px;
        }

        .checkbox-container {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-top: 8px;
        }

        .checkbox-container input[type="checkbox"] {
            width: auto;
        }

        .main-content {
            flex: 1;
            display: flex;
            flex-direction: column;
        }

        .header {
            background: var(--secondary-bg);
            padding: 20px;
            border-bottom: 2px solid var(--text-primary);
            display: flex;
            align-items: center;
            gap: 15px;
        }
        
        .header h1 {
            font-size: 24px;
            font-weight: bold;
            color: var(--text-primary);
            letter-spacing: 2px;
            text-transform: uppercase;
        }

        .header p {
            color: var(--text-secondary);
            font-size: 14px;
        }

        .chat-container {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            background: var(--primary-bg);
        }

        .message {
            margin-bottom: 20px;
            animation: fadeIn 0.3s;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .message-header {
            font-weight: bold;
            margin-bottom: 5px;
            font-size: 14px;
            color: var(--text-primary);
        }

        .user-message .message-header {
            color: var(--text-primary);
        }

        .assistant-message .message-header {
            color: var(--text-primary);
        }

        .system-message .message-header {
            color: var(--text-secondary);
            font-style: italic;
        }

        .message-content {
            padding: 12px;
            border-radius: 8px;
            background: var(--secondary-bg);
            border: 1px solid var(--border-color);
            line-height: 1.6;
            white-space: pre-wrap;
        }
        
        .user-message .message-content {
            border-left: 3px solid var(--text-primary);
        }
        
        .assistant-message .message-content {
            border-left: 3px solid var(--text-secondary);
            background: var(--input-bg);
        }
        
        .system-message .message-content {
            border-left: 3px solid var(--accent-secondary);
            font-style: italic;
            color: var(--text-secondary);
        }

        .thinking {
            color: var(--text-secondary);
            font-style: italic;
            animation: pulse 1.5s infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        .quick-commands {
            padding: 10px 20px;
            background: var(--secondary-bg);
            border-top: 1px solid var(--border-color);
            display: flex;
            gap: 10px;
        }

        .input-area {
            padding: 20px;
            background: var(--secondary-bg);
            border-top: 1px solid var(--border-color);
            display: flex;
            gap: 10px;
            align-items: center;
        }

        .input-field {
            flex: 1;
            padding: 12px;
            background: var(--input-bg);
            color: var(--text-primary);
            border: 2px solid var(--border-color);
            border-radius: 6px;
            font-size: 14px;
            outline: none;
            transition: border-color 0.3s;
        }
        
        .input-field:focus {
            border-color: var(--text-primary);
        }

        .send-btn {
            padding: 12px 30px;
            background: var(--text-primary);
            color: var(--primary-bg);
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-weight: bold;
            font-size: 14px;
            transition: all 0.3s;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .send-btn:hover {
            background: var(--text-secondary);
            transform: scale(1.02);
        }

        .send-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .status-bar {
            padding: 8px 20px;
            background: var(--sidebar-bg);
            border-top: 1px solid var(--border-color);
            font-size: 12px;
            color: var(--text-secondary);
        }

        ::-webkit-scrollbar {
            width: 10px;
        }

        ::-webkit-scrollbar-track {
            background: var(--primary-bg);
        }

        ::-webkit-scrollbar-thumb {
            background: var(--border-color);
            border-radius: 5px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: #555;
        }
    </style>
</head>
<body>
    <div class="container">
        <aside class="sidebar">
            <div class="sidebar-title">SETTINGS</div>

            <div class="settings-section">
                <label class="settings-label">Theme:</label>
                <div class="theme-buttons">
                    <button class="btn" onclick="setTheme('dark')">Dark</button>
                    <button class="btn" onclick="setTheme('light')">Light</button>
                </div>
            </div>

            <div class="settings-section">
                <label class="settings-label">Font Size:</label>
                <div class="font-buttons">
                    <button class="btn" onclick="adjustFontSize(-1)">A-</button>
                    <button class="btn" onclick="adjustFontSize(1)">A+</button>
                </div>
            </div>

            <div class="settings-section">
                <label class="settings-label">Response Length:</label>
                <select id="responseLength" onchange="updateSetting('responseLength', this.value)">
                    <option value="short">Short</option>
                    <option value="medium" selected>Medium</option>
                    <option value="long">Long</option>
                </select>
            </div>

            <div class="settings-section">
                <label class="settings-label">Memory:</label>
                <div class="checkbox-container">
                    <input type="checkbox" id="sessionMemory" checked onchange="updateSetting('sessionMemory', this.checked)">
                    <label for="sessionMemory">Session Memory</label>
                </div>
                <div class="checkbox-container">
                    <input type="checkbox" id="persistentMemory" onchange="updateSetting('persistentMemory', this.checked)">
                    <label for="persistentMemory">Persistent Memory</label>
                </div>
            </div>

            <div class="settings-section">
                <label class="settings-label">Quick Actions:</label>
                <button class="btn btn-full" onclick="clearChat()">Clear Chat</button>
                <button class="btn btn-full" onclick="exportChat()">Export Chat</button>
                <button class="btn btn-full" onclick="clearMemory()">Clear Memory</button>
            </div>
        </aside>

        <main class="main-content">
            <header class="header">
                <div>
                    <h1>R3KON GPT</h1>
                    <p>Professional Cybersecurity Assistant</p>
                </div>
            </header>

            <div class="chat-container" id="chatContainer">
                <div class="message system-message">
                    <div class="message-header">System</div>
                    <div class="message-content">Loading R3KON GPT model... Please wait.</div>
                </div>
            </div>

            <div class="quick-commands">
                <button class="btn" onclick="quickCommand('summarize')">Summarize</button>
                <button class="btn" onclick="quickCommand('explain')">Explain Simply</button>
            </div>

            <div class="input-area">
                <input 
                    type="text" 
                    id="userInput" 
                    class="input-field" 
                    placeholder="Type your message here..."
                    onkeypress="handleKeyPress(event)"
                    disabled
                >
                <button class="send-btn" id="sendBtn" onclick="sendMessage()" disabled>Send</button>
            </div>

            <div class="status-bar" id="statusBar">Initializing...</div>
        </main>
    </div>

    <script>
        let config = {
            fontSize: 14,
            theme: 'dark',
            responseLength: 'medium',
            sessionMemory: true,
            persistentMemory: false
        };

        let conversationHistory = [];
        let sessionMemory = [];
        let modelLoaded = false;

        function loadConfig() {
            const saved = localStorage.getItem('rekon_config');
            if (saved) {
                config = { ...config, ...JSON.parse(saved) };
                applyConfig();
            }
        }

        function saveConfig() {
            localStorage.setItem('rekon_config', JSON.stringify(config));
        }

        function applyConfig() {
            document.body.style.fontSize = config.fontSize + 'px';
            if (config.theme === 'light') {
                document.body.classList.add('light-theme');
            }
            document.getElementById('responseLength').value = config.responseLength;
            document.getElementById('sessionMemory').checked = config.sessionMemory;
            document.getElementById('persistentMemory').checked = config.persistentMemory;
        }

        function setTheme(theme) {
            config.theme = theme;
            if (theme === 'light') {
                document.body.classList.add('light-theme');
            } else {
                document.body.classList.remove('light-theme');
            }
            saveConfig();
        }

        function adjustFontSize(delta) {
            config.fontSize = Math.max(10, Math.min(20, config.fontSize + delta));
            document.body.style.fontSize = config.fontSize + 'px';
            saveConfig();
        }

        function updateSetting(key, value) {
            config[key] = value;
            saveConfig();
        }

        function addMessage(sender, content, type = 'user') {
            const chatContainer = document.getElementById('chatContainer');
            const timestamp = new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
            
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${type}-message`;
            
            const header = document.createElement('div');
            header.className = 'message-header';
            header.textContent = `${sender} [${timestamp}]:`;
            
            const contentDiv = document.createElement('div');
            contentDiv.className = 'message-content';
            contentDiv.textContent = content;
            
            messageDiv.appendChild(header);
            messageDiv.appendChild(contentDiv);
            chatContainer.appendChild(messageDiv);
            
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }

        function showThinking() {
            const chatContainer = document.getElementById('chatContainer');
            const thinkingDiv = document.createElement('div');
            thinkingDiv.id = 'thinking';
            thinkingDiv.className = 'message thinking';
            thinkingDiv.innerHTML = '<div class="message-content">R3KON GPT is thinking...</div>';
            chatContainer.appendChild(thinkingDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;
            
            updateStatus('Generating response...');
        }

        function removeThinking() {
            const thinking = document.getElementById('thinking');
            if (thinking) thinking.remove();
            updateStatus('Ready');
        }

        function updateStatus(text) {
            document.getElementById('statusBar').textContent = text;
        }

        async function sendMessage() {
            const input = document.getElementById('userInput');
            const message = input.value.trim();
            
            if (!message || !modelLoaded) return;
            
            addMessage('You', message, 'user');
            conversationHistory.push({ role: 'user', content: message });
            
            input.value = '';
            input.disabled = true;
            document.getElementById('sendBtn').disabled = true;
            
            showThinking();
            
            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        message: message,
                        config: config,
                        history: sessionMemory
                    })
                });
                
                const data = await response.json();
                removeThinking();
                
                if (data.response) {
                    addMessage('R3KON GPT', data.response, 'assistant');
                    
                    if (config.sessionMemory) {
                        sessionMemory.push({ user: message, assistant: data.response });
                        if (sessionMemory.length > 8) sessionMemory.shift();
                    }
                } else {
                    addMessage('Error', 'Failed to get response from model.', 'system');
                }
            } catch (error) {
                removeThinking();
                addMessage('Error', `Connection error: ${error.message}`, 'system');
            }
            
            input.disabled = false;
            document.getElementById('sendBtn').disabled = false;
            input.focus();
        }

        function handleKeyPress(event) {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                sendMessage();
            }
        }

        function quickCommand(type) {
            const commands = {
                summarize: 'Summarize your last response in 2-3 bullet points.',
                explain: 'Explain your last response in simpler terms.'
            };
            
            if (conversationHistory.length < 2) {
                alert('Please have a conversation first.');
                return;
            }
            
            document.getElementById('userInput').value = commands[type];
            sendMessage();
        }

        function clearChat() {
            if (confirm('Clear chat history?')) {
                const chatContainer = document.getElementById('chatContainer');
                chatContainer.innerHTML = '';
                conversationHistory = [];
                sessionMemory = [];
                addMessage('System', 'Chat cleared. Ready for new conversation.', 'system');
            }
        }

        function exportChat() {
            const chatContainer = document.getElementById('chatContainer');
            const messages = chatContainer.innerText;
            const blob = new Blob([messages], { type: 'text/plain' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `rekon_chat_${Date.now()}.txt`;
            a.click();
            URL.revokeObjectURL(url);
        }

        function clearMemory() {
            if (confirm('Clear all memory?')) {
                sessionMemory = [];
                localStorage.removeItem('rekon_memory');
                alert('Memory cleared successfully.');
            }
        }

        async function initialize() {
            loadConfig();
            
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                
                if (data.modelLoaded) {
                    modelLoaded = true;
                    addMessage('System', 'Model loaded successfully! Ask me anything about cybersecurity.', 'system');
                    document.getElementById('userInput').disabled = false;
                    document.getElementById('sendBtn').disabled = false;
                    updateStatus('Ready');
                    document.getElementById('userInput').focus();
                } else {
                    addMessage('System', 'Model failed to load. Please check the console.', 'system');
                    updateStatus('Error: Model not loaded');
                }
            } catch (error) {
                addMessage('System', 'Cannot connect to backend. Make sure the server is running.', 'system');
                updateStatus('Error: Backend not connected');
            }
        }

        window.onload = initialize;
    </script>
</body>
</html>'''
    
    return html_content

@app.route('/api/status')
def status():
    """Check if model is loaded"""
    return jsonify({
        "modelLoaded": model_loaded,
        "status": "ready" if model_loaded else "loading"
    })

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat requests"""
    if not model_loaded:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        data = request.json
        message = data.get('message', '')
        config = data.get('config', {})
        history = data.get('history', [])
        
        if not message:
            return jsonify({"error": "No message"}), 400
        
        result = generate_response(message, config, history)
        return jsonify(result)
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return jsonify({"error": str(e)}), 500

def start_flask(port):
    """Start Flask server in background"""
    global flask_started
    try:
        print(f"Starting Flask server on port {port}...")
        load_model()
        flask_started = True
        app.run(host='127.0.0.1', port=port, debug=False, use_reloader=False, threaded=True)
    except Exception as e:
        print(f"ERROR: Flask failed to start: {e}")
        import traceback
        traceback.print_exc()
        flask_started = False

def wait_for_flask(port, timeout=30):
    """Wait for Flask to be ready"""
    import urllib.request
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            urllib.request.urlopen(f'http://127.0.0.1:{port}/api/status', timeout=1)
            print("Flask server is ready!")
            return True
        except:
            time.sleep(0.5)
    
    return False

def main():
    """Main entry point for desktop app"""
    # Set UTF-8 encoding for console (only if stdout exists)
    if sys.platform == 'win32':
        try:
            if hasattr(sys.stdout, 'reconfigure'):
                sys.stdout.reconfigure(encoding='utf-8')
                sys.stderr.reconfigure(encoding='utf-8')
        except:
            pass
    
    print("=" * 60)
    print("R3KON GPT - Desktop Application")
    print("=" * 60)
    
    # Find a free port
    port = find_free_port()
    print(f"Using port: {port}")
    
    # Start Flask in background thread
    print("Starting backend server...")
    flask_thread = threading.Thread(target=start_flask, args=(port,), daemon=True)
    flask_thread.start()
    
    # Wait for Flask to be ready
    print("Waiting for server to start...")
    if not wait_for_flask(port, timeout=30):
        print("ERROR: Server failed to start within 30 seconds")
        print("\nTroubleshooting:")
        print("1. Check if model file exists in 'model' folder")
        print("2. Make sure llama-cpp-python is installed")
        print("3. Check console for error messages above")
        # Don't use input() - just wait and exit
        time.sleep(5)
        return
    
    print("Server started successfully!")
    print(f"Opening window at http://127.0.0.1:{port}")
    
    # Create desktop window
    try:
        window = webview.create_window(
            'R3KON GPT ',
            f'http://127.0.0.1:{port}',
            width=1200,
            height=800,
            resizable=True,
            fullscreen=False,
            min_size=(900, 600),
        )
        
        print("Window created!")
        webview.start()
        
    except Exception as e:
        print(f"ERROR: Failed to create window: {e}")
        import traceback
        traceback.print_exc()
        # Don't use input() - just wait and exit
        time.sleep(5)

if __name__ == '__main__':
    main()