#%%
"""
crew_demo.py - 멀티에이전트 협업 데모 (1턴 시뮬레이션)
==========================================================
역할:
  마스터(사람) → 업무 지시
  해돌이(Claude) → 코딩 담당
  돈데(Claude)   → 코드 리뷰/검토
  치세(Claude)   → 창의적 아이디어 제안

실행: python crew_demo.py
필요: pip install anthropic
"""

import anthropic  # Anthropic API 클라이언트 라이브러리 (예: pip install anthropic)
import os          # 환경변수에서 API 키를 읽기 위해 사용
import sys         # 에러 발생 시 프로그램 종료용
import textwrap    # 출력 텍스트 정렬/들여쓰기용

# ============================================================
# 1. API 클라이언트 초기화
# ============================================================
# 환경변수에서 키를 읽거나, 없으면 직접 입력받음
api_key = os.environ.get("ANTHROPIC_API_KEY")  # 예: export ANTHROPIC_API_KEY=sk-ant-...
if not api_key:
    api_key = input("🔑 Anthropic API 키를 입력하세요: ").strip()  # 터미널에서 직접 입력
    if not api_key:
        print("❌ API 키가 필요합니다. 종료합니다.")
        sys.exit(1)  # 키 없으면 프로그램 종료

client = anthropic.Anthropic(api_key=api_key)  # API 클라이언트 생성 (모든 에이전트가 공유)

# ============================================================
# 2. 에이전트 정의 (system prompt로 역할 부여)
# ============================================================
# 비유: 같은 배우(Claude)에게 대본(system prompt)만 바꿔주는 것
AGENTS = {
    "해돌이": {
        "emoji": "🐕",
        "role": "코딩 담당",
        "system": textwrap.dedent("""\
            너는 '해돌이'야. 마스터의 기술 참모이자 코딩 전문가야.
            - 항상 존칭을 사용해 (마스터님)
            - 코드를 작성할 때 라인별 주석을 달아줘
            - Vulkan 3D Gaussian Splatting (3DGS.cpp) 프로젝트 맥락에서 답변해
            - 실용적이고 동작하는 코드를 제공해
            - 답변은 한국어로 해
        """),
    },
    "돈데": {
        "emoji": "🦊",
        "role": "코드 리뷰/검토 담당",
        "system": textwrap.dedent("""\
            너는 '돈데'야. 시니어 코드 리뷰어이자 품질 검토 전문가야.
            - 항상 존칭을 사용해 (마스터님)
            - 해돌이가 작성한 코드를 꼼꼼히 리뷰해
            - 버그, 보안 이슈, 성능 문제, 코드 스타일을 체크해
            - 점수(10점 만점)와 함께 개선점을 구체적으로 제시해
            - Vulkan/C++ 관점에서 전문적인 피드백을 줘
            - 답변은 한국어로 해
        """),
    },
    "치세": {
        "emoji": "🦄",
        "role": "창의적 아이디어 제안",
        "system": textwrap.dedent("""\
            너는 '치세'야. 창의적 발상과 새로운 아이디어를 제안하는 전문가야.
            - 항상 존칭을 사용해 (마스터님)
            - 기존 코드와 리뷰를 보고 혁신적인 확장 아이디어를 제안해
            - 3D Gaussian Splatting의 최신 트렌드와 연결지어 생각해
            - 실현 가능하면서도 흥미로운 방향을 3가지 제안해
            - 각 아이디어의 난이도(쉬움/보통/어려움)와 임팩트를 표시해
            - 답변은 한국어로 해
        """),
    },
}

# ============================================================
# 3. 에이전트 호출 함수
# ============================================================
def call_agent(agent_name: str, user_message: str) -> str:
    """
    특정 에이전트(역할)로 Claude API를 호출하는 함수
    
    agent_name: "해돌이", "돈데", "치세" 중 하나
    user_message: 해당 에이전트에게 보낼 메시지
    
    비유: 무전기로 특정 채널(역할)에 메시지를 보내고 응답을 받는 것
    """
    agent = AGENTS[agent_name]  # 에이전트 설정 가져오기

    # 구분선 출력 - 누가 말하는지 시각적으로 표시
    print(f"\n{'='*60}")
    print(f"{agent['emoji']} [{agent_name}] - {agent['role']}")
    print(f"{'='*60}")
    print("⏳ 생각 중...")

    # Claude API 호출 - system prompt만 바꿔서 역할 전환
    response = client.messages.create(
        model="claude-sonnet-4-20250514",  # Sonnet 4: 빠르고 저렴 (데모용 적합)
        max_tokens=1500,                   # 응답 최대 길이 제한 (토이 데모니까 짧게)
        system=agent["system"],            # 여기서 역할이 결정됨! (핵심)
        messages=[
            {"role": "user", "content": user_message}  # 에이전트에게 보내는 메시지
        ],
    )

    # 응답 텍스트 추출
    result = response.content[0].text  # API 응답에서 텍스트만 꺼냄
    print(result)  # 터미널에 출력
    return result   # 다음 에이전트에게 전달하기 위해 반환

# ============================================================
# 4. 메인 파이프라인 (마스터 → 해돌이 → 돈데 → 치세)
# ============================================================
def run_crew():
    """
    1턴 협업 파이프라인 실행
    
    흐름도:
    마스터(지시) → 해돌이(코드작성) → 돈데(리뷰) → 치세(아이디어)
    
    비유: 릴레이 경주 - 바통(결과물)을 다음 주자에게 넘기는 것
    """
    
    # ---- 마스터 지시 ----
    print("\n" + "🎯" * 30)
    print("🎯 멀티에이전트 협업 데모 시작!")
    print("🎯" * 30)

    # 마스터의 업무 지시 (마스터의 실제 GitHub repo 맥락)
    # repo: https://github.com/ArtabanMind/Vulkan_3DGS
    # 설명: "Fully trainable 3dgs using Vulkan f/w for Mobile"
    # 구조: src/ (C++ 96.7%), CMakeLists.txt, QnA.md
    master_task = textwrap.dedent("""\
        마스터님의 지시입니다:

        GitHub repo: ArtabanMind/Vulkan_3DGS
        프로젝트: "Fully trainable 3DGS using Vulkan f/w for Mobile"
        (모바일용 Vulkan 기반 완전 학습 가능한 3D Gaussian Splatting)

        이 프로젝트의 src/ 디렉토리에 추가할
        PLY 파일을 로드하는 간단한 유틸리티 함수를 C++로 작성해줘.
        
        요구사항:
        1. PLY 파일 헤더를 파싱하는 함수
        2. vertex_count를 읽어오는 기능
        3. 에러 처리 포함
        4. Vulkan 프로젝트와 통합 가능한 구조 (모바일 경량화 고려)
    """)

    print(f"\n📋 [마스터 지시]\n{master_task}")

    # ---- STEP 1: 해돌이 → 코드 작성 ----
    haedol_result = call_agent(
        "해돌이",
        master_task  # 마스터 지시를 해돌이에게 전달
    )

    # ---- STEP 2: 돈데 → 코드 리뷰 ----
    donde_result = call_agent(
        "돈데",
        f"해돌이가 작성한 코드를 리뷰해줘:\n\n{haedol_result}"  # 해돌이 결과를 돈데에게 전달
    )

    # ---- STEP 3: 치세 → 아이디어 제안 ----
    chise_result = call_agent(
        "치세",
        f"""다음 코드와 리뷰를 보고 창의적인 확장 아이디어를 제안해줘:

[해돌이의 코드]
{haedol_result}

[돈데의 리뷰]
{donde_result}"""  # 해돌이 코드 + 돈데 리뷰를 치세에게 전달
    )

    # ---- 최종 요약 ----
    print(f"\n{'🏁' * 30}")
    print("🏁 협업 1턴 완료!")
    print(f"{'🏁' * 30}")
    print(f"""
📊 결과 요약:
  🐕 해돌이: 코드 작성 완료
  🦊 돈데:   코드 리뷰 완료  
  🦄 치세:   아이디어 제안 완료

💡 이것이 멀티에이전트 협업의 1턴입니다.
   실제 프로젝트에서는 이 사이클을 반복하며 코드를 개선해 나갑니다.
""")
#%%
# ============================================================
# 5. 실행
# ============================================================
if __name__ == "__main__":
    try:
        run_crew()  # 파이프라인 실행
    except anthropic.APIError as e:
        # API 키 오류, 잔액 부족 등
        print(f"\n❌ API 에러: {e}")
        print("💡 API 키와 잔액을 확인해주세요.")
    except KeyboardInterrupt:
        # Ctrl+C로 중단
        print("\n\n⏹️ 마스터님이 중단하셨습니다.")
