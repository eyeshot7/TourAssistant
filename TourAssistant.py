import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# --- Azure OpenAI 모델 초기화 ---
try:
    # 여행지 추천 및 상세 정보 생성을 위한 고성능 모델
    # llm = AzureChatOpenAI(
    #     deployment_name=os.getenv("AZURE_OPENAI_GPT4_1_DEPLOYMENT"),  # GPT-4.1 모델
    # )

    llm = AzureChatOpenAI(
        deployment_name=os.getenv("AZURE_OPENAI_GPT4_1_MINI_DEPLOYMENT"),  # GPT-4.1 Mini 모델
    )
except Exception as e:
    st.error(f"Azure OpenAI 모델 초기화 중 오류가 발생했습니다: {e}")
    st.stop()

# --- MBTI 질문 ---
mbti_questions = [
    {
        "question": "에너지 충전이 필요할 때, 당신은 보통 어떻게 하시나요?",
        "options": {
            "E": "사람들과 어울리며 활기찬 시간을 보낸다.",
            "I": "조용한 곳에서 혼자만의 시간을 보내며 재충전한다."
        }
    },
    {
        "question": "새로운 정보를 접할 때, 당신은 어떤 방식에 더 끌리나요?",
        "options": {
            "S": "실제 경험과 오감으로 느낄 수 있는 구체적인 사실에 집중한다.",
            "N": "숨겨진 의미와 가능성을 상상하며 전체적인 그림을 본다."
        }
    },
    {
        "question": "결정을 내릴 때, 당신에게 더 중요한 기준은 무엇인가요?",
        "options": {
            "T": "논리적이고 객관적인 분석을 통해 공정하게 판단한다.",
            "F": "상황에 관련된 사람들의 감정과 관계를 우선적으로 고려한다."
        }
    },
    {
        "question": "여행을 계획할 때, 당신의 스타일은 어떤가요?",
        "options": {
            "J": "미리 계획을 세우고, 정해진 일정에 따라 움직이는 것을 선호한다.",
            "P": "상황에 따라 유연하게 계획을 변경하며, 즉흥적인 결정을 즐긴다."
        }
    }
]

# --- Streamlit 세션 ---
if "page" not in st.session_state:
    st.session_state.page = "start"
if "mbti_type" not in st.session_state:
    st.session_state.mbti_type = ""
if "answers" not in st.session_state:
    st.session_state.answers = []
if "recommendations" not in st.session_state:
    st.session_state.recommendations = []
if "previous_recommendations" not in st.session_state:
    st.session_state.previous_recommendations = []

def get_mbti_description(mbti):
    """GPT-4.1-mini를 사용하여 MBTI 유형에 대한 간단한 설명을 생성합니다."""
    messages = [
        SystemMessage(content="You are a helpful AI assistant that provides a brief and easy-to-understand description of an MBTI type in Korean."),
        HumanMessage(content=f"{mbti} 유형에 대해 2-3문장으로 간단하게 설명해줘.")
    ]
    try:
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        return f"MBTI 설명을 가져오는 중 오류 발생: {e}"

def get_travel_recommendations(mbti, previous_recs):
    """GPT-4.1-mini를 사용하여 MBTI에 맞는 여행지 3곳을 추천받습니다."""
    previous_recs_str = ", ".join(previous_recs) if previous_recs else "없음"
    prompt = (
        f"{mbti} 유형의 사람이 좋아할 만한 새로운 여행지 3곳을 추천해줘. "
        "각 여행지는 '1. [여행지 이름]' 형식으로 번호를 붙여서 제시하고, 왜 추천하는지 한 문장으로 간략하게 설명해줘. "
        f"이전에 추천했던 여행지({previous_recs_str})는 제외하고 완전히 새로운 곳으로 추천해줘."
    )
    messages = [
        SystemMessage(content="You are a travel expert who recommends personalized travel destinations based on MBTI types."),
        HumanMessage(content=prompt)
    ]
    try:
        response = llm.invoke(messages)
        return response.content.strip().split('\n')
    except Exception as e:
        return [f"여행지 추천을 받는 중 오류 발생: {e}"]


def get_destination_details(destination):
    """GPT-4.1-mini를 사용하여 선택된 여행지의 상세 정보와 일정을 생성합니다."""
    prompt = (
        f"'{destination}' 여행지에 대한 상세한 정보와 3일간의 간략한 여행 일정을 추천해줘.\n\n"
        "### 상세 정보:\n"
        "- 이 여행지의 매력과 특징을 3-4문장으로 설명해줘.\n"
        "- 꼭 방문해야 할 명소 2-3곳을 알려줘.\n\n"
        "### 3일 여행 일정 예시:\n"
        "- **1일차:** [오전/오후 활동]\n"
        "- **2일차:** [오전/오후 활동]\n"
        "- **3일차:** [오전/오후 활동]\n\n"
        "전체적으로 마크다운 형식을 사용해서 보기 좋게 작성해줘."
    )
    messages = [
        SystemMessage(content="You are a detailed travel guide that provides comprehensive information and itineraries."),
        HumanMessage(content=prompt)
    ]
    try:
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        return f"상세 정보를 가져오는 중 오류 발생: {e}"

# --- UI 렌더링 ---

st.title("✈️ MBTI 맞춤 여행 추천 챗봇")
st.markdown("Azure OpenAI와 LangChain을 사용하여 당신의 MBTI에 꼭 맞는 특별한 여행지를 찾아보세요!")

# --- 페이지 라우팅 ---

# 시작 페이지
if st.session_state.page == "start":
    st.info("당신의 MBTI를 알고 계신가요?")
    knows_mbti = st.radio("", ("네, 알고 있어요.", "아니요, 잘 모르겠어요."), key="knows_mbti", horizontal=True)

    if knows_mbti == "네, 알고 있어요.":
        mbti_input = st.text_input("당신의 MBTI를 4자리 영문 대문자로 입력해주세요. (예: INFP)").upper()
        if st.button("확인"):
            if len(mbti_input) == 4 and all(c in "IE" or c in "SN" or c in "TF" or c in "JP" for c in mbti_input):
                st.session_state.mbti_type = mbti_input
                st.session_state.page = "recommend"
                st.rerun()
            else:
                st.error("올바른 MBTI 형식이 아닙니다. 다시 입력해주세요.")
    else:
        if st.button("MBTI 진단 시작하기"):
            st.session_state.page = "mbti_test"
            st.rerun()


# MBTI 진단 페이지
elif st.session_state.page == "mbti_test":
    question_index = len(st.session_state.answers)
    if question_index < len(mbti_questions):
        q = mbti_questions[question_index]
        st.subheader(f"질문 {question_index + 1}/{len(mbti_questions)}")
        st.write(q["question"])
        
        # 버튼을 가로로 배치하기 위해 columns 사용
        cols = st.columns(len(q["options"]))
        for i, (mbti_char, option_text) in enumerate(q["options"].items()):
            if cols[i].button(option_text, key=f"q{question_index}_{mbti_char}"):
                st.session_state.answers.append(mbti_char)
                st.rerun()
    else:
        st.session_state.mbti_type = "".join(st.session_state.answers)
        st.session_state.page = "mbti_result"
        st.rerun()

# MBTI 결과 및 설명 페이지
elif st.session_state.page == "mbti_result":
    st.success(f"당신의 MBTI는 **{st.session_state.mbti_type}** 입니다!")
    with st.spinner("MBTI 유형에 대한 설명을 생성 중입니다..."):
        description = get_mbti_description(st.session_state.mbti_type)
        st.markdown(description)
    
    if st.button("나에게 맞는 여행지 추천받기"):
        st.session_state.page = "recommend"
        st.rerun()

# 여행지 추천 페이지
elif st.session_state.page == "recommend":
    if not st.session_state.recommendations:
        with st.spinner(f"{st.session_state.mbti_type} 유형에 맞는 여행지를 찾고 있습니다..."):
            recommendations = get_travel_recommendations(st.session_state.mbti_type, st.session_state.previous_recommendations)
            st.session_state.recommendations = recommendations
            # 이전 추천 목록에 현재 추천 목록 추가
            st.session_state.previous_recommendations.extend([rec.split('. ')[1].split(' -')[0] for rec in recommendations if '. ' in rec])


    if st.session_state.recommendations:
        st.subheader(f"🌍 {st.session_state.mbti_type}님을 위한 맞춤 여행지 추천")
        for rec in st.session_state.recommendations:
            st.write(rec)

        st.markdown("---")
        selection = st.selectbox("가장 마음에 드는 여행지의 번호를 선택하세요.", [""] + [str(i) for i in range(1, 4)])
        if selection:
            try:
                # 선택된 여행지 이름 추출 (예: '1. 파리 - 로맨틱한 예술의 도시'에서 '파리' 추출)
                selected_dest_name = st.session_state.recommendations[int(selection)-1].split('. ')[1].split(' -')[0]
                st.session_state.selected_destination = selected_dest_name
                st.session_state.page = "details"
                st.rerun()
            except (IndexError, ValueError):
                st.error("잘못된 선택입니다. 1, 2, 3 중에서 선택해주세요.")

        if st.button("마음에 드는 곳이 없어요 (다른 여행지 추천)"):
            st.session_state.recommendations = []
            st.rerun()


# 상세 정보 페이지
elif st.session_state.page == "details":
    destination = st.session_state.selected_destination
    st.header(f"✨ {destination} 여행 정보")
    
    with st.spinner(f"'{destination}'의 상세 정보를 불러오는 중입니다..."):
        details = get_destination_details(destination)
        st.markdown(details)
    
    if st.button("다른 추천 여행지 보기"):
        st.session_state.page = "recommend"
        st.session_state.selected_destination = None
        st.rerun()