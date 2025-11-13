# 분석서비스를 제공하는 .py파일
"""
면접 종료(s3) 엔드포인트용 비동기 분석 서비스

이 서비스는 다음을 수행한다:
의도(Intent) 점수 계산 — 모든 문장의 결과를 기반으로 5대 NCS 역량(의사소통, 직무수행능력, 성실성, 적응력, 리더십/팀워크)의 평균 점수를 산출한다.
감정(Emotion) 라벨 분류 — 각 문장별로 감정을 분류하며, 불확실=0, 긍정=1, 부정=2 로 표시한다.
종합 리포트 생성 — 위 점수와 감정 라벨, 그리고 LLM 분석 결과를 결합해 전체 보고서 텍스트를 생성한다.

이 버전은 비동기 환경에서 안전하게 동작한다:
- 모델들은 한 번만 로드되어 전역적으로 캐시된다.
- 모델 로딩과 추론(inference)은 이벤트 루프를 막지 않기 위해 스레드 풀(Executor)에서 수행된다.
- 여러 개의 요청이 동시에 들어와도, 동일한 로드된 모델을 안전하게 공유한다.
"""

import logging # 로그 남기기 위한 모듈
from typing import Dict, Any, List, Tuple # 
from pathlib import Path # 파일 경로나 폴더를 다룰 때 문자열 대신 객체지향적 방식으로 다루는 클래스
import os # 운영체제와 상호작용하는 기본 모듈
import asyncio # 비동기처리를 위한 라이브러리
from concurrent.futures import ThreadPoolExecutor

# anay
logger = logging.getLogger(__name__)

# Global Cache & lock for async-safe lazy loading
_MODEL_CACHE : Dict[str, Any] = {}
_MODEL_LOCK = asyncio.Lock()
_EXECUTOR: ThreadPoolExecutor | None = None

# 비동기 서버 환경에서 CPU연산을 안전하게 돌리기 위한 스레드 풀 초기화 함수
def _get_executor() -> ThreadPoolExecutor : 
    global _EXECUTOR
    if _EXECUTOR is None : 
        _EXECUTOR = ThreadPoolExecutor(max_workers=int(os.getenv("ANALYSIS_WORKERS", "4")))
    return _EXECUTOR

# 모델이 있는 파일을 가져온다.
def _load_models_sync() -> Dict[str, Any] :
    """Synchronous part: import and load models + tokenizer.
    Returns a dict containing intent_model, emotion_model module refs, tokenizer, label_map.
    """
    from . import intent_model as im
    from . import emotion_model as emo
    # 모델 파일 경로를 안정적으로 찾기 위함
    repo_root = Path(__file__).resolve().parents[2]
    intent_model_path = os.getenv(
        "INTENT_MODEL_PATH",
        str(repo_root / "ai" / "v1_code" / "using_custom_models" / "model_intent_quantized.pt")
    )
    intent_label_map_path = os.getenv(
        "INTENT_LABEL_MAP_PATH",
        str(repo_root / "ai" / "v1_code" / "using_custom_models" / "label_map.txt")
    )
    tokenizer = im.get_tokenizer()
    model, label_map = im.load_quantize_model(intent_model_path, intent_label_map_path)
    return {
        "intent_model" : model,
        "intent_label_map" : label_map,
        "tokenizer" : tokenizer,
        "intent_mod" : im,
        "emotion_mod" : emo,
    }

# 모델 캐시를 비동기 환경에서도 안전하게 한 번만 로드하는 핵심 함수
# 
async def _ensure_models_loaded() -> Dict[str, Any] : 
    """Async-safe lazy loader. Ensures models are loaded exactly once."""
    # model을 매번 새로 불러오지 않고 한 번만 로드해서 계속 써먹는다.
    if _MODEL_CACHE:
        return _MODEL_CACHE
    async with _MODEL_LOCK : 
        if _MODEL_CACHE :
            return _MODEL_CACHE
        loop = asyncio.get_event_loop()
        start = loop.time()
        cache = await loop.run_in_executor(_get_executor(), _load_models_sync)
        _MODEL_CACHE.update(cache)
        dur = loop.time() - start
        try : 
            labels_preview = list(cache["intent_label_map"].values()) if isinstance(cache["intent_label_map"], dict) else cache["intent_label_map"]
        except Exception : 
            labels_preview = "<unknown labels>"
        logger.info(f"Analysis models loaded in {dur:.3f}s : intent labels ={labels_preview}")
        return _MODEL_CACHE

# 면접 답변(qna history)기반 의도와 감정을 동시에 분석
def _compute_scores_sync(cache: Dict[str, Any], qna_history: List[Dict[str, str]]) -> Tuple[Dict[str, int], List[int]] : 
    """Synchronous CPU-bound inference logic. Run in thread pool."""
    im = cache["intent_mod"]
    emo = cache["emotion_mod"]
    model = cache["intent_model"]
    tokenizer = cache["tokenizer"]
    label_map = cache["intent_label_map"]
    
    intent_probs_sum = {label_map[i]: 0.0 for i in range(len(label_map))}
    intent_count = 0
    emotion_labels: List[int] = []

    for qa in qna_history : 
        answer = (qa.get("answer") or "").strip()
        if not answer : 
            continue
        sentences = emo.split_sentences(answer)
        for sent in sentences : 
            if not sent : 
                continue

            # Intent
            try :
                pred_label, conf, probs = im.predict_intent(model, tokenizer, sent, label_map)
                for idx, prob_val in enumerate(probs) : 
                    intent_probs_sum[label_map[idx]] += float(prob_val)
                intent_count += 1
            # 예측 실패시
            except Exception as e :
                logger.warning(f"Intent prediction failed for sentence: {e}")

            # Emotion
            try : 
                # predict_sentences? 이건 어떤 메서드인거지
                sent_results, _ = emo.predict_sentences([sent])
                if sent_results : 
                    emo_label_str = sent_results[0].get("pred_label", "uncertain")
                    emo_int = {"uncertain": 0, "positive": 1, "negative": 2}.get(emo_label_str, 0)
                    emotion_labels.append(emo_int)

            except Exception as e : 
                logger.warning(f"Emotion prediction failed for sentence: {e}")
    scores = Dict[str, int] = {}
    if intent_count > 0 : 
        for comp, total_prob in intent_probs_sum.items() : 
            avg_prob = total_prob / intent_count
            scores[comp] = int(round(avg_prob * 100))
    else : 
        for comp in intent_probs_sum.keys() : 
            scores[comp] = 0
    logger.info(f"Analysis: computed intent scores from {intent_count} sentences; {len(emotion_labels)} emotion labels")
    return scores, emotion_labels

# 
async def compute_intent_scores_and_emotion_labels(qna_history: List[Dict[str, str]]) -> tuple[Dict[str, int], List[int]]:
    try : 
        cache = await _ensure_models_loaded()
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_get_executor(), _compute_scores_sync, cache, qna_history)
    except Exception as e :
        logger.exception(f"Failed to compute intent/emotion analysis: {e}")
        default_scores = {
            "Communication": 0,
            "Teamwork_Leadership": 0,
            "Integrity": 0,
            "Adaptability": 0,
            "Job_Competency": 0,
        }
        return default_scores, []

# 보고서 LLM 프롬프팅과 로직들
def generate_final_report(
        scores : Dict[str, int],
        emotion_labels : List[int],
        qna_history: List[Dict[str, str]],
        jd_text : str
) -> str :
    """
    Generate a textual report combining scores, emotion labels, and LLM analysis.
    
    Returns a string summary suitable for the 'report' field.
    """
    try : 
        # Build prompt for LLM
        scores_text = "\n".join(f"- {k}: {v}/100" for k, v in scores.items())

        emotion_summary = f"{len(emotion_labels)} sentences analyzed"
        if emotion_labels : 
            emo_counts = {"uncertain" : 0, "positive" : 0, "negative" : 0}
            for emo_int in emotion_labels : 
                emo_label = {0: "uncertain", 1: "positive", 2 : "negative"}.get(emo_int, "uncertain")
                emo_counts[emo_label] += 1
            emotion_summary += f" (positive={emo_counts['positive']}, neutral={emo_counts['uncertain']}, negative={emo_counts['negative']})"

        qna_text = "\n".join(
            f"Q: {qa.get('question', '')}\nA: {qa.get('answer', '')}"
            for qa in qna_history
        )

        # 기준
        avg_score = sum(scores.values()) / len(scores) if scores else 0
        high_scores = [k for k, v in scores.items() if v >= 70]
        low_scores = [k for k, v in scores.items() if v < 40]

        emo_counts = {"uncertain": 0, "positive": 0, "negative": 0}
        if emotion_labels:
            for emo_int in emotion_labels:
                emo_label = {0: "uncertain", 1: "positive", 2: "negative"}.get(emo_int, "uncertain")
                emo_counts[emo_label] += 1
        
        total_sents = sum(emo_counts.values())
        positive_ratio = emo_counts["positive"] / total_sents if total_sents > 0 else 0
        negative_ratio = emo_counts["negative"] / total_sents if total_sents > 0 else 0

        performance_note = ""
        if avg_score >= 70:
            performance_note = "전반적으로 높은 역량 점수를 보임 (평균 70점 이상). 강점을 중심으로 긍정적 평가."
        elif avg_score >= 50:
            performance_note = "전반적으로 중간 수준의 역량 점수 (평균 50-70점). 균형잡힌 평가와 개선 방향 제시."
        else:
            performance_note = "전반적으로 낮은 역량 점수 (평균 50점 미만). 건설적인 피드백과 구체적 개선 방안 제시."
        
        emotion_note = ""
        if positive_ratio >= 0.6:
            emotion_note = "긍정적 감정 비율이 높음 (60% 이상). 자신감과 열정을 반영."
        elif negative_ratio >= 0.3:
            emotion_note = "부정적 감정 비율이 다소 높음 (30% 이상). 긴장도나 불안감 고려."
        else:
            emotion_note = "감정적으로 중립적이고 안정적인 답변."

        # prompting 세팅어
        system_prompt = (
            "당신은 NCS(국가직무능력표준) 기반 면접 평가 전문가입니다. "
            "역량 점수, 감정 분석, 직무 설명을 종합하여 전문적이고 건설적인 면접 평가 보고서를 한국어로 작성합니다.\n\n"
            "평가 컨텍스트:\n"
            "- 지원자는 신입/newcomer로 지원한 후보자입니다\n"
            "- 평가 기준은 신입 수준에 맞춰져야 하며, 잠재력과 학습 능력을 중시합니다\n"
            "- 광범위한 실무 경험보다는 기본 역량, 태도, 성장 가능성을 평가합니다\n\n"
            "평가 원칙:\n"
            "- 객관적이고 구체적이어야 하며, 실행 가능한 피드백을 제공해야 합니다\n"
            "- 신입 수준에서의 강점과 개선점을 명확히 구분합니다\n"
            "- 즉시 투입 가능 여부보다는 교육 후 성장 가능성에 초점을 맞춥니다"
        )

        user_prompt = f"""다음 데이터를 바탕으로 종합적인 면접 평가 보고서를 작성하세요:
**NCS 역량 점수 (0-100):**
{scores_text}
평균 점수: {avg_score:.1f}
강점 역량: {', '.join(high_scores) if high_scores else '없음'}
개선 필요 역량: {', '.join(low_scores) if low_scores else '없음'}

**감정 분석:**
{emotion_summary}
긍정 비율: {positive_ratio:.1%}, 부정 비율: {negative_ratio:.1%}

**평가 가이드:**
- {performance_note}
- {emotion_note}

**직무 설명 (JD):**
{jd_text[:500]}

**면접 Q&A:**
{qna_text[:2000]}

다음 항목을 포함하여 작성하되, 450-500자로 적어줘.
**종합 평가**: 역량 점수와 감정 분석을 종합하여 지원자의 전반적인 면접 수행 평가
단, 다음 내용을 추가해서
1. **주요 강점**: 높은 점수를 받은 역량과 긍정적 측면
2. **개선 영역**: 낮은 점수를 받은 역량과 보완이 필요한 부분
3. **직무 적합성**: JD와 역량 점수를 고려한 적합도
4. **최종 추천**: 다음 단계 진행 여부 및 구체적 제안
"""
        
        # Try using LLM with proper message structure
        try:
            from . import question_model as qm
            llm = qm._get_openai_chat()
            if llm is None:
                raise RuntimeError("Chat model unavailable")
            
            # Use chat message format
            from langchain.schema import SystemMessage, HumanMessage
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = llm(messages)
            report = response.content if hasattr(response, 'content') else str(response)
            logger.info("Generated final report using LLM")
            return report
        except Exception as e:
            logger.warning(f"LLM report generation failed, using fallback: {e}")
            # Fallback: basic text report
            fallback_report = f"""
면접 평가 보고서

**역량 점수:**
{scores_text}

**감정 분석:**
{emotion_summary}

**종합 평가:**
지원자는 전반적으로 안정적인 답변을 제공했습니다. 
각 역량 점수를 참고하여 추가 검토가 필요합니다.

**추천사항:**
다음 단계로 진행 가능합니다.
"""
            return fallback_report
            
    except Exception as e:
        logger.exception(f"Failed to generate final report: {e}")
        return "Report generation failed. Please review raw data."
        