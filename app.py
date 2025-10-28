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
