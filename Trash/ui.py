import streamlit as st

st.set_page_config(page_title="Roadmap", layout="wide")

st.markdown(
    """
<style>
    :root {
        --primary: #6366f1;
        --secondary: #8b5cf6;
        --accent: #ec4899;
        --bg-dark: #0f172a;
        --bg-card: #1e293b;
        --border: #334155;
        --text-primary: #f1f5f9;
        --text-secondary: #cbd5e1;
        --success: #10b981;
    }
    
    * {
        margin: 0;
        padding: 0;
    }
    
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, var(--bg-dark) 0%, #1a1f3a 100%);
        min-height: 100vh;
    }
    
    [data-testid="stMainBlockContainer"] {
        padding: 60px 40px;
    }
    
    .roadmap-wrapper {
        max-width: 1000px;
        margin: 0 auto;
    }
    
    .roadmap-header {
        margin-bottom: 80px;
        text-align: center;
    }
    
    .roadmap-title {
        font-size: 48px;
        font-weight: 700;
        background: linear-gradient(135deg, #6366f1 0%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 16px;
        letter-spacing: -1px;
    }
    
    .roadmap-subtitle {
        font-size: 16px;
        color: var(--text-secondary);
    }
    
    .stage-section {
        margin-bottom: 100px;
        position: relative;
    }
    
    .stage-header {
        display: flex;
        align-items: center;
        gap: 16px;
        margin-bottom: 40px;
    }
    
    .stage-number {
        width: 44px;
        height: 44px;
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        color: white;
        font-size: 18px;
        flex-shrink: 0;
        box-shadow: 0 8px 24px rgba(99, 102, 241, 0.3);
    }
    
    .stage-title {
        font-size: 28px;
        font-weight: 600;
        color: var(--text-primary);
        letter-spacing: -0.5px;
    }
    
    .stage-arrow {
        text-align: center;
        font-size: 32px;
        color: var(--primary);
        margin: 60px 0;
        opacity: 0.6;
        animation: float 3s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(10px); }
    }
    
    .nodes-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 20px;
        margin-bottom: 20px;
    }
    
    .node-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 24px;
        cursor: pointer;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .node-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--primary), var(--secondary));
        transform: scaleX(0);
        transform-origin: left;
        transition: transform 0.3s ease;
    }
    
    .node-card:hover {
        border-color: var(--primary);
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.05) 0%, rgba(139, 92, 246, 0.05) 100%);
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(99, 102, 241, 0.2);
    }
    
    .node-card:hover::before {
        transform: scaleX(1);
    }
    
    .node-status {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 12px;
        padding: 4px 10px;
        border-radius: 6px;
        width: fit-content;
    }
    
    .status-planning {
        background: rgba(99, 102, 241, 0.15);
        color: #a5b4fc;
    }
    
    .status-in-progress {
        background: rgba(236, 72, 153, 0.15);
        color: #f472b6;
    }
    
    .status-completed {
        background: rgba(16, 185, 129, 0.15);
        color: #6ee7b7;
    }
    
    .node-title {
        font-size: 18px;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 10px;
        line-height: 1.4;
    }
    
    .node-description {
        font-size: 13px;
        color: var(--text-secondary);
        line-height: 1.6;
        margin-bottom: 16px;
    }
    
    .node-meta {
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
        font-size: 12px;
    }
    
    .meta-tag {
        padding: 4px 8px;
        background: rgba(99, 102, 241, 0.1);
        color: #a5b4fc;
        border-radius: 4px;
        border: 1px solid rgba(99, 102, 241, 0.2);
    }
    
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--border), transparent);
        margin: 80px 0;
    }
    
    /* Streamlit overrides */
    .stExpander {
        background: transparent !important;
        border: none !important;
    }
    
    .stExpander > div > button {
        background: transparent !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="roadmap-wrapper">', unsafe_allow_html=True)

# Header
st.markdown(
    """
<div class="roadmap-header">
    <div class="roadmap-title">Product Roadmap 2025</div>
    <div class="roadmap-subtitle">Strategic vision and development timeline</div>
</div>
""",
    unsafe_allow_html=True,
)

# Roadmap data
ROADMAP = {
    "Q1: Foundation": [
        {
            "title": "Project Setup",
            "desc": "Initialize repository, CI/CD, and dev tools",
            "status": "completed",
            "tags": ["infrastructure", "backend"],
        },
        {
            "title": "Core Architecture",
            "desc": "Design system patterns and interfaces",
            "status": "completed",
            "tags": ["architecture", "design"],
        },
        {
            "title": "Authentication",
            "desc": "User login, registration, and security",
            "status": "in-progress",
            "tags": ["auth", "security"],
        },
    ],
    "Q2: MVP Features": [
        {
            "title": "User Dashboard",
            "desc": "Main interface and profile management",
            "status": "in-progress",
            "tags": ["frontend", "ui"],
        },
        {
            "title": "API Integration",
            "desc": "RESTful endpoints and request handling",
            "status": "planning",
            "tags": ["backend", "api"],
        },
        {
            "title": "Database Layer",
            "desc": "Data models, queries, and optimization",
            "status": "planning",
            "tags": ["database", "backend"],
        },
    ],
    "Q3: Enhancement": [
        {
            "title": "Analytics Dashboard",
            "desc": "Data visualization and reporting tools",
            "status": "planning",
            "tags": ["analytics", "frontend"],
        },
        {
            "title": "Performance Tuning",
            "desc": "Caching, compression, and optimization",
            "status": "planning",
            "tags": ["performance", "backend"],
        },
        {
            "title": "Advanced Search",
            "desc": "Full-text search and filtering capabilities",
            "status": "planning",
            "tags": ["search", "feature"],
        },
    ],
    "Q4: Launch": [
        {
            "title": "Security Audit",
            "desc": "Penetration testing and compliance check",
            "status": "planning",
            "tags": ["security", "qa"],
        },
        {
            "title": "Production Deploy",
            "desc": "Final deployment and monitoring setup",
            "status": "planning",
            "tags": ["devops", "launch"],
        },
    ],
}

status_colors = {
    "completed": ("completed", "✓ Completed", "status-completed"),
    "in-progress": ("in-progress", "🔄 In Progress", "status-in-progress"),
    "planning": ("planning", "📋 Planning", "status-planning"),
}

# Render roadmap
stages = list(ROADMAP.keys())
for stage_idx, stage_name in enumerate(stages):
    # Stage header
    st.markdown(
        f"""
    <div class="stage-header">
        <div class="stage-number">{stage_idx + 1}</div>
        <div class="stage-title">{stage_name}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Render nodes in grid
    st.markdown('<div class="nodes-grid">', unsafe_allow_html=True)

    for node in ROADMAP[stage_name]:
        status_key, status_label, status_class = status_colors[node["status"]]

        st.markdown(
            f"""
        <div class="node-card">
            <div class="node-status {status_class}">
                <span>{status_label}</span>
            </div>
            <div class="node-title">{node['title']}</div>
            <div class="node-description">{node['desc']}</div>
            <div class="node-meta">
                {''.join([f'<span class="meta-tag">{tag}</span>' for tag in node['tags']])}
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # Arrow between stages
    if stage_idx < len(stages) - 1:
        st.markdown('<div class="stage-arrow">↓</div>', unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
