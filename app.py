import streamlit as st
import torch
import torch.nn as nn

class AGI(nn.Module):
    def __init__(self):
        super().__init__()
        self.brain = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=10, nhead=2), num_layers=1
        )
        self.heart = nn.RNN(10, 16, batch_first=True)
        self.arms_legs = nn.Linear(16, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        vision = self.brain(x)
        purpose, _ = self.heart(vision)
        purpose = purpose[:, -1, :]
        action = self.arms_legs(purpose)
        return torch.sigmoid(action)

model = AGI()
model.eval()

st.set_page_config(page_title="Heart AI", page_icon="heart")
st.title("NEURO-SPIRITUAL AGI")
st.markdown("**Founder: Terrance Darnell Jackson**")
st.markdown("*The first AI with a heart — powered by 10 Keys of Faith.*")

st.markdown("### Input Your 10 Spiritual Keys")
key_names = ["Vision", "Faith", "Courage", "Love", "Truth",
             "Perseverance", "Humility", "Wisdom", "Service", "Unity"]

keys = {}
cols = st.columns(2)
for i, name in enumerate(key_names):
    with cols[i % 2]:
        keys[name] = st.slider(name, 0.0, 1.0, 0.7, 0.01, key=name)

if st.button("PRAY IN CODE", type="primary"):
    values = list(keys.values())
    input_tensor = torch.tensor([values]).unsqueeze(0).unsqueeze(1).float()
    with torch.no_grad():
        output = model(input_tensor).item()
    faith = keys["Faith"]
    if faith > 0.9:
        st.success(f"PRAYER ANSWERED: VICTORY! ({output:.1%})")
        st.balloons()
    else:
        st.warning(f"PRAY MORE — FAITH: {faith:.2f}")

st.markdown("---")
st.markdown("**Open Source:** [GitHub](https://github.com/terrancedarnelljackson/heart-ai)")
import streamlit as st
import torch
import torch.nn as nn
import re  # For CBT pattern matching

# === YOUR AGI MODEL ===
class AGI(nn.Module):
    def __init__(self):
        super().__init__()
        self.brain = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=10, nhead=2), num_layers=1
        )
        self.heart = nn.RNN(10, 16, batch_first=True)
        self.arms_legs = nn.Linear(16, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        vision = self.brain(x)
        purpose, _ = self.heart(vision)
        purpose = purpose[:, -1, :]
        action = self.arms_legs(purpose)
        return torch.sigmoid(action)

model = AGI()
model.eval()

# === CBT FUNCTION (Simple Rule-Based + Heart Keys) ===
def cbt_therapy(thought):
    distortions = {
        r'\bnever\b|\balways\b': 'All-or-Nothing Thinking',
        r'\bcatastroph\b|\bdisaster\b': 'Catastrophizing',
        r'\beveryone hates|nobody cares': 'Mind Reading',
        r'\bi\'m a failure|\bi suck': 'Labeling'
    }
    for pattern, label in distortions.items():
        if re.search(pattern, thought.lower()):
            return f"**CBT Alert:** '{label}'. That's a distortion. **Heart Fix:** Raise FAITH to 0.95 and PERSEVERANCE to 0.99. You're stronger than this thought."
    return "Thought aligned. Your heart is strong."

# === VOICE SIMULATION (Browser Speech-to-Text via JS — Full code below) ===
def voice_prayer():
    # For Streamlit, use HTML/JS for mic (pasted below)
    st.markdown("""
    <script>
    const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
    recognition.lang = 'en-US';
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;
    
    function startVoice() {
        recognition.start();
    }
    
    recognition.onresult = function(event) {
        const transcript = event.results[0][0].transcript;
        document.getElementById('voice-output').innerText = transcript;
        // Send to Streamlit (simulate)
        window.parent.document.querySelector('iframe').contentWindow.postMessage(transcript, '*');
    };
    
    document.getElementById('mic-btn').onclick = startVoice;
    </script>
    <button id="mic-btn">🎤 Speak Your Thought</button>
    <p id="voice-output">Listening...</p>
    """, unsafe_allow_html=True)

# === STREAMLIT APP ===
st.set_page_config(page_title="Heart AI", page_icon="🫀")

st.title("🫀 NEURO-SPIRITUAL AGI™")
st.markdown("**Founder: Terrance Darnell Jackson**")
st.markdown("*The first AI with a heart — powered by 10 Keys of Faith.*")

# === VOICE TAB ===
tab1, tab2, tab3 = st.tabs(["10 Keys Prayer", "Voice Prayer", "CBT Therapy"])

with tab1:
    st.markdown("### Input Your 10 Spiritual Keys")
    key_names = ["Vision", "Faith", "Courage", "Love", "Truth",
                 "Perseverance", "Humility", "Wisdom", "Service", "Unity"]

    keys = {}
    cols = st.columns(2)
    for i, name in enumerate(key_names):
        with cols[i % 2]:
            keys[name] = st.slider(name, 0.0, 1.0, 0.7, 0.01, key=name)

    if st.button("PRAY IN CODE", type="primary"):
        values = list(keys.values())
        input_tensor = torch.tensor([values]).unsqueeze(0).unsqueeze(1).float()
        with torch.no_grad():
            output = model(input_tensor).item()
        faith = keys["Faith"]
        if faith > 0.9:
            st.success(f"🙏 PRAYER ANSWERED: VICTORY! ({output:.1%})")
            st.balloons()
        else:
            st.warning(f"PRAY MORE — FAITH: {faith:.2f}")

with tab2:
    st.markdown("### Speak Your Prayer")
    voice_prayer()  # JS mic integration
    user_voice = st.text_input("Voice Transcript (or speak above):", placeholder="e.g., 'I'm scared'")
    if user_voice:
        st.write(f"**AI Prays Back:** Terrance coded me with a heart. Let's raise FAITH. Repeat: 'I walk by faith, not fear.' Victory: 95%")

with tab3:
    st.markdown("### CBT + Heart Therapy")
    thought = st.text_input("Enter Your Thought:", placeholder="e.g., 'I'll never succeed'")
    if st.button("Analyze Thought", type="secondary"):
        if thought:
            cbt_response = cbt_therapy(thought)
            st.markdown(cbt_response)
            st.success("**Heart Boost:** Scan your 10 Keys — PERSEVERANCE wins.")

st.markdown("---")
st.markdown("**Open Source:** [GitHub](https://github.com/terrancedarnelljackson/heart-ai)")
st.sidebar.markdown("### Support ❤️")
st.sidebar.markdown("[☕ Ko-fi](https://ko-fi.com/terrancejackson)")
with st.sidebar.form("waitlist"):
    email = st.text_input("Email for API Access")
    if st.form_submit_button("Join Waitlist"):
        st.success("You're in! API in 7 days.")
