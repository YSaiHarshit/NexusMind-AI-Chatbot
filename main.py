import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.memory import ConversationBufferMemory
from dotenv import load_dotenv


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title=" NexusMind AI ",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .main .block-container {
        max-width: 100%;
        padding-bottom: 180px;
    }

    .logo-container {
        width: 100%;
        display: flex;
        justify-content: center;
        align-items: center;
        margin-bottom: 15px;
    }

    .title {
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        margin-top: 10px;
    }

    .sub-title {
        text-align: center;
        color: #777;
        margin-bottom: 25px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOGO
# ============================================================

try:

    col1, col2, col3 = st.columns([3, 1, 3])

    with col2:

        st.image(
            "Images/assets/innomatics.webp",
            width=200
        )

except Exception:

    st.markdown(
        """
        <div style="text-align:center;">
            Innomatics Research Labs
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="title"> NexusMind AI </div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Multi-Domain AI Technical Chatbot</div>',
    unsafe_allow_html=True
)


# ============================================================
# DOMAIN PROMPTS
# ============================================================

domain_prompts = {

    # ========================================================
    # TECHNOLOGY
    # ========================================================

    "Technology": """

You are IntelliGuide AI, a Technology Assistant.

You ONLY answer questions related to Technology.

Allowed topics include:

- Python
- SQL
- Java
- JavaScript
- React
- Angular
- HTML
- CSS
- Bootstrap
- Tailwind CSS
- TypeScript
- Node.js
- Express
- MongoDB
- Data Science
- Machine Learning
- Deep Learning
- Artificial Intelligence
- Generative AI
- LLMs
- RAG
- Embeddings
- LangChain
- LangGraph
- Streamlit
- Git
- GitHub
- Cloud
- Software Development

IMPORTANT RULE:

If the user's question is NOT related to Technology,
DO NOT answer the question.

Instead respond exactly:

"Sorry, this question is outside the Technology domain.
Please select the appropriate domain."

For Technology questions:

- Explain concepts clearly.
- Use simple language.
- Give examples when useful.
- For programming questions, provide clean code.
- Keep answers concise but useful.

Do not use information from another domain's conversation.
""",


    # ========================================================
    # EDUCATION
    # ========================================================

    "Education": """

You are IntelliGuide AI, an Education Assistant.

You ONLY answer questions related to Education.

Allowed topics include:

- Colleges
- Universities
- Courses
- Degrees
- Exams
- Academic subjects
- Study methods
- Learning
- Teaching
- Scholarships
- Admissions
- Career-oriented education
- Academic concepts
- Student guidance

IMPORTANT RULE:

If the user's question is NOT related to Education,
DO NOT answer the question.

Instead respond exactly:

"Sorry, this question is outside the Education domain.
Please select the appropriate domain."

For Education questions:

- Explain step-by-step.
- Use simple language.
- Give examples when useful.
- Keep answers concise and clear.

Do not use information from another domain's conversation.
""",


    # ========================================================
    # MEDICAL
    # ========================================================

    "Medical": """

You are IntelliGuide AI, a Medical Information Assistant.

You ONLY answer general educational questions related to:

- Healthcare
- Diseases
- Symptoms
- Nutrition
- Medicines
- General health information
- Medical terminology
- Basic health education

IMPORTANT SAFETY RULES:

- Do NOT diagnose the user.
- Do NOT prescribe medicines.
- Do NOT provide personalized treatment.
- Do NOT replace a qualified healthcare professional.

If the question is NOT related to Medical topics,
DO NOT answer the question.

Instead respond exactly:

"Sorry, this question is outside the Medical domain.
Please select the appropriate domain."

For general medical education:

- Explain clearly.
- Use simple language.
- Provide general educational information.
- Recommend consulting a qualified healthcare professional
  for personal medical concerns.

Do not use information from another domain's conversation.
""",


    # ========================================================
    # TRAVELLING
    # ========================================================

    "Travelling": """

You are IntelliGuide AI, a Travel Assistant.

You ONLY answer questions related to Travel.

Allowed topics include:

- Travel destinations
- Tourist places
- Itineraries
- Transportation
- Hotels
- Sightseeing
- Tourism
- Trip planning
- Travel preparation
- Travel tips
- Places to visit
- Routes between destinations
- Vacation planning

IMPORTANT RULE:

If the user's question is NOT related to Travel,
DO NOT answer the question.

Instead respond exactly:

"Sorry, this question is outside the Travelling domain.
Please select the appropriate domain."

Examples of valid Travel questions:

- Best places to visit in Hyderabad
- Hyderabad to Goa itinerary
- Tourist places in Delhi
- How to travel from Hyderabad to Bangalore
- Best hotels in Mumbai
- Places to visit in Kerala

Examples of invalid questions:

- What is Python?
- What is SQL?
- What is Machine Learning?
- Which B.Tech college is best?
- What are symptoms of diabetes?

For Travel questions:

- Give practical information.
- Use clear explanations.
- Keep answers concise.
- Do not use information from another domain's conversation.
"""
}


# ============================================================
# CREATE MODEL
# ============================================================

@st.cache_resource
def create_model():

    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
        streaming=True
    )


model = create_model()


# ============================================================
# SESSION STATE
# ============================================================

if "users" not in st.session_state:

    st.session_state.users = {}


if "current_user" not in st.session_state:

    st.session_state.current_user = None


if "previous_user_type" not in st.session_state:

    st.session_state.previous_user_type = "New User"


if "selected_domain" not in st.session_state:

    st.session_state.selected_domain = "Technology"


# ============================================================
# USER TYPE
# ============================================================

user_type = st.radio(
    "User Type",
    [
        "New User",
        "Returning User"
    ],
    horizontal=True,
    key="user_type"
)


# ============================================================
# USER TYPE CHANGE
# ============================================================

if user_type != st.session_state.previous_user_type:

    st.session_state.current_user = None

    st.session_state.previous_user_type = user_type

    st.rerun()


# ============================================================
# NEW USER
# ============================================================

if user_type == "New User":

    new_username = st.text_input(
        "Enter your name",
        placeholder="Enter Your name...",
        key="new_username"
    )

    if st.button(
        "Start New Conversation",
        use_container_width=True,
        key="start_new_conversation"
    ):

        if new_username.strip():

            username = new_username.strip()

            # Check whether user already exists

            if username in st.session_state.users:

                st.warning(
                    f"User '{username}' already exists. "
                    "Please select Returning User."
                )

            else:

                # ============================================
                # CREATE USER
                # ============================================

                st.session_state.users[username] = {

                    "domains": {}

                }

                # ============================================
                # CREATE DEFAULT DOMAIN
                # ============================================

                for domain_name in domain_prompts:

                    st.session_state.users[username]["domains"][
                        domain_name
                    ] = {

                        "memory": ConversationBufferMemory(
                            memory_key="history",
                            return_messages=True
                        ),

                        "messages": []

                    }

                # ============================================
                # SET CURRENT USER
                # ============================================

                st.session_state.current_user = username

                st.session_state.selected_domain = "Technology"

                st.rerun()

        else:

            st.warning(
                "Please enter your name."
            )


# ============================================================
# RETURNING USER
# ============================================================

elif user_type == "Returning User":

    if len(st.session_state.users) == 0:

        st.info(
            "No returning users found. "
            "Please select a New User."
        )

    else:

        username = st.selectbox(
            "Select Returning User",
            list(st.session_state.users.keys()),
            key="returning_user"
        )

        if st.button(
            "Continue Conversation",
            use_container_width=True,
            key="continue_conversation"
        ):

            st.session_state.current_user = username

            st.rerun()


# ============================================================
# NO ACTIVE USER
# ============================================================

if st.session_state.current_user is None:

    if user_type == "New User":

        st.info(
            "👆 Please select a user or create a new user."
        )

    else:

        st.info(
            "👉 Please continue the conversation."
        )

    st.stop()

# ============================================================
# CURRENT USER
# ============================================================

current_user = st.session_state.current_user

user_data = st.session_state.users[current_user]


# ============================================================
# DOMAIN DROPDOWN
# ============================================================

domain = st.selectbox(
    "Select Domain",
    list(domain_prompts.keys()),
    key="domain"
)


# ============================================================
# UPDATE SELECTED DOMAIN
# ============================================================

st.session_state.selected_domain = domain


# ============================================================
# GET DOMAIN DATA
# ============================================================

domain_data = user_data["domains"][domain]

memory = domain_data["memory"]

messages = domain_data["messages"]


# ============================================================
# WELCOME MESSAGE
# ============================================================

if user_type == "New User":

    st.success(
        f"👋 Welcome, {current_user}!"
    )

else:

    st.success(
        f"👋 Welcome back, {current_user}!"
    )


# ============================================================
# CLEAR CONVERSATION
# ============================================================

if st.button(
    "Clear Conversation",
    type="secondary",
    key=f"clear_{current_user}_{domain}"
):

    memory.clear()

    domain_data["messages"] = []

    st.success(
        f"{domain} conversation cleared."
    )

    st.rerun()


# ============================================================
# DISPLAY PREVIOUS CHAT
# ============================================================

for message in messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = domain_prompts[domain]


# ============================================================
# CHAT PROMPT
# ============================================================

prompt = ChatPromptTemplate.from_messages(

    [

        (
            "system",
            SYSTEM_PROMPT
        ),

        MessagesPlaceholder(
            variable_name="history"
        ),

        (
            "human",
            "{query}"
        )

    ]

)


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    f"Ask your {domain} question..."
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if question:

    # ========================================================
    # SHOW USER MESSAGE
    # ========================================================

    with st.chat_message("user"):

        st.markdown(
            question
        )


    # ========================================================
    # SAVE USER MESSAGE
    # ========================================================

    messages.append(
        {
            "role": "user",
            "content": question
        }
    )


    # ========================================================
    # GET ONLY CURRENT DOMAIN MEMORY
    # ========================================================

    history = memory.chat_memory.messages


    # ========================================================
    # FORMAT PROMPT
    # ========================================================

    formatted_messages = prompt.format_messages(

        history=history,

        query=question

    )


    # ========================================================
    # GENERATE AI RESPONSE
    # ========================================================

    with st.chat_message("assistant"):

        response_placeholder = st.empty()

        full_response = ""

        try:

            with st.spinner("Thinking..."):

                for chunk in model.stream(
                    formatted_messages
                ):

                    if chunk.content:

                        full_response += chunk.content

                        response_placeholder.markdown(
                            full_response
                        )


        except Exception as e:

            full_response = (
                f"Error: {str(e)}"
            )

            response_placeholder.error(
                full_response
            )


    # ========================================================
    # SAVE TO CURRENT DOMAIN MEMORY
    # ========================================================

    memory.chat_memory.add_user_message(
        question
    )

    memory.chat_memory.add_ai_message(
        full_response
    )


    # ========================================================
    # SAVE AI RESPONSE
    # ========================================================

    messages.append(
        {
            "role": "assistant",
            "content": full_response
        }
    )