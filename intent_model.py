import torch
import torch.nn as nn
from kobert_transformers import get_tokenizer
import re

# ⚠️ 양자화 모델은 CPU 전용!
device = torch.device("cpu")


class BertClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size: int = 768,
                 num_classes: int = 5,
                 dr_rate: float = 0.3,
                 class_weights: torch.Tensor | None = None):
        super().__init__()
        self.bert = bert
        self.dropout = nn.Dropout(p=dr_rate) if dr_rate and dr_rate > 0 else nn.Identity()
        self.classifier = nn.Linear(hidden_size, num_classes)

        # 🔁 BCEWithLogitsLoss 사용 (Sigmoid 기반 멀티라벨 학습용)
        # class_weights가 있을 경우 BCEWithLogitsLoss의 pos_weight로 사용
        if class_weights is not None:
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weights)
        else:
            self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor | None = None,
                token_type_ids: torch.Tensor | None = None,
                labels: torch.Tensor | None = None):

        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)

        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            # 마지막 hidden state의 [CLS] 토큰 사용
            pooled = outputs[0][:, 0]

        logits = self.classifier(self.dropout(pooled))

        if labels is not None:
            # BCEWithLogitsLoss는 float 타겟 기대
            loss = self.loss_fn(logits, labels.float())
            return logits, loss
        return logits, None


def load_quantized_model(model_path, label_map_path):
    """양자화된 모델과 라벨 매핑 로드 (CPU 전용)"""

    print("⚠️  양자화 모델은 CPU에서만 실행됩니다.")
    print("💡 GPU를 사용하려면 원본 모델(model.pt)을 사용하세요.\n")

    # 라벨 매핑 로드 (idx \t label)
    label_map = {}
    with open(label_map_path, 'r', encoding='utf-8') as f:
        for line in f:
            idx, label = line.strip().split('\t')
            label_map[int(idx)] = label

    print("양자화 모델 로드 중...")

    import sys as _sys
    main_mod = _sys.modules.get('__main__')
    if main_mod is not None:
        # 🔴 저장 당시에 KoBertClassifier 이름으로 저장됐을 수도 있어서 alias 맞춰줌
        for name in ("KoBertClassifier", "BertClassifier"):
            if not hasattr(main_mod, name):
                setattr(main_mod, name, BertClassifier)

    # FutureWarning은 우리가 신뢰하는 자기 파일이니까 그냥 weights_only=False 명시
    try:
        model = torch.load(model_path, map_location='cpu', weights_only=False)
    except TypeError:
        # Torch 버전에 따라 weights_only 인자가 없을 수도 있어서 fallback
        model = torch.load(model_path, map_location='cpu')

    model.eval()
    print("✓ 모델 로드 완료 (CPU 모드)\n")
    return model, label_map


def split_sentences(text: str):
    """문단을 문장으로 분리"""
    # 줄바꿈/여러 공백 제거
    text = re.sub(r'\s+', ' ', text.strip())

    # 문장 분리 (마침표, 물음표, 느낌표 기준 + 공백)
    sentences = re.split(r'(?<=[.!?])\s+', text)

    # 빈 문장 제거
    sentences = [s.strip() for s in sentences if s.strip()]

    return sentences


@torch.no_grad()
def predict_intent(model, tokenizer, sentence, label_map, max_len: int = 502):
    """
    단일 문장의 의도 예측
    - Sigmoid + BCEWithLogitsLoss로 학습된 멀티라벨 모델을
      "Top-1 + Top-2 역량" 형태로 해석
    """
    model.eval()

    # 토크나이징
    encoding = tokenizer(
        sentence,
        padding='max_length',
        truncation=True,
        max_length=max_len,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    token_type_ids = encoding.get('token_type_ids')

    # 예측
    logits, _ = model(input_ids, attention_mask, token_type_ids)

    # 🔁 BCEWithLogitsLoss 기반이므로 softmax가 아니라 sigmoid 사용
    # logits: [1, num_classes] → probs: [num_classes]
    probs = torch.sigmoid(logits)[0]  # shape: (num_classes,)

    # 🔝 Top-2 인덱스와 값
    topk_vals, topk_idx = torch.topk(probs, k=2)
    topk_vals = topk_vals.tolist()
    topk_idx = topk_idx.tolist()

    # 1등 정보 (기존과 동일)
    pred_idx = topk_idx[0]
    confidence = topk_vals[0]

    # Top-2를 (label, prob) 리스트로 구성
    top2 = [(label_map[i], v) for i, v in zip(topk_idx, topk_vals)]

    return label_map[pred_idx], confidence, probs.numpy(), top2


def analyze_paragraph(model, tokenizer, paragraph: str, label_map):
    """문단 전체 분석"""
    sentences = split_sentences(paragraph)

    print("=" * 80)
    print("📝 문단 분석 결과")
    print("=" * 80)
    print(f"\n총 {len(sentences)}개의 문장이 발견되었습니다.\n")

    results = []

    for i, sentence in enumerate(sentences, 1):
        intent, confidence, probs, top2 = predict_intent(model, tokenizer, sentence, label_map)
        results.append({
            'sentence_num': i,
            'sentence': sentence,
            'intent': intent,
            'confidence': confidence,
            'probabilities': probs,
            'top2': top2,
        })

        print(f"[문장 {i}]")
        print(f"내용: {sentence}")
        print(f"분류(Top-1): {intent} (확신도: {confidence:.2%})")

        # 🔝 Top-2 역량 요약 출력
        print("Top-2 역량:")
        for rank, (lbl, val) in enumerate(top2, 1):
            print(f"  {rank}) {lbl}: {val:.2%}")

        # 전체 세부 확률 출력
        print("세부 확률:")
        for idx, prob in enumerate(probs):
            print(f"  - {label_map[idx]}: {prob:.2%}")
        print("-" * 80)

    return results


def print_summary(results):
    """요약 통계 출력"""
    print("\n" + "=" * 80)
    print("📊 분석 요약")
    print("=" * 80)

    # Top-1 기준 문장 수 집계
    intent_counts = {}
    for result in results:
        intent = result['intent']
        intent_counts[intent] = intent_counts.get(intent, 0) + 1

    print("\n의도별 문장 수 (Top-1 기준):")
    for intent, count in sorted(intent_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(results)) * 100
        print(f"  {intent}: {count}개 ({percentage:.1f}%)")

    avg_confidence = sum(r['confidence'] for r in results) / len(results)
    print(f"\n평균 확신도(Top-1): {avg_confidence:.2%}")


# 메인 실행 코드
if __name__ == "__main__":
    print("=" * 80)
    print("🚀 양자화 KoBERT 모델 추론 (CPU 최적화, Sigmoid + BCEWithLogitsLoss)")
    print("=" * 80)
    print()

    # 🔁 Windows 경로는 raw string(r"...") 사용 권장
    MODEL_PATH = r"C:\Users\SSAFY\Desktop\VR_interview\model_intent_v2_quantized.pt"
    LABEL_MAP_PATH = r"C:\Users\SSAFY\Desktop\VR_interview\label_map.txt"

    print("모델을 로드하는 중...")
    tokenizer = get_tokenizer()
    model, label_map = load_quantized_model(MODEL_PATH, LABEL_MAP_PATH)

    # 분석할 문단
    sentence = """
가장 몰입했던 경험은 시각장애인을 위한 자율주행 RC카 개발 프로젝트를 진행했던 것입니다.
당시 WebRTC 라이브러리를 이용하여 보호자 대시보드에 RC카가 영상을 실시간으로 송신하고자 하였습니다. 하지만 해당 라이브러리 사양의 한계로 인해, 카메라 영상을 직접적으로 받을 수 없었습니다. 이에 접근 방식을 영상 전송에서 이미지 프레임 전송으로 변경하였고, 좌표 메타데이터로 AI 객체 분석 결과를 전송하려던 부분을 이미지 프레임 하나로 함께 전송할 수 있게 되었습니다. 결과적으로 보호자 대시보드에서는 객체 탐지 박스가 포함된 이미지를 연속적으로 표시하여 영상처럼 보이도록 만들 수 있었습니다.
동시에 AI 객체 탐지를 위해서 데이터를 학습해야 했는데 실내 이용이 가능함을 MVP로 만들고자 하였기에 의자나 책상, 가방, 사람, 벽과 같은 데이터를 주로 이용하였습니다. 하지만 같은 의자라고 해도 보여지는 시점에 따라 이미지가 완전히 달라지기 때문에 많은 데이터가 오히려 정확도를 떨어뜨리는 결과가 나왔습니다. 이에 데이터를 다시금 정비하고, 직접 사진을 찍어 데이터를 준비하는 등의 접근을 취하자 정확도를 향상시킬 수 있었습니다.
이러한 경험들을 통해 라이브러리 이용 시에는 사전에 사양을 확실하게 파악하는 것이 중요하고, AI에서는 많은 데이터가 항상 좋지는 않다는 점을 배웠으며 이는 이후 업무를 효율적으로 진행할 수 있도록 만들 것입니다.
"""

    # 분석 실행
    results = analyze_paragraph(model, tokenizer, sentence, label_map)

    # 요약 출력
    print_summary(results)

    print("\n" + "=" * 80)
    print("💡 팁:")
    print("  - 양자화 모델은 원본 대비 용량이 작고 추론 속도가 빠릅니다")
    print("  - CPU에서 최적화되어 있어 GPU 없이도 빠른 추론 가능")
    print("  - FastAPI 등 프로덕션 배포에 적합합니다")
    print("=" * 80)
