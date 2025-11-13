import re, numpy as np, torch
from kobert_transformers import get_tokenizer
import torch.nn as nn

DEVICE  = torch.device("cpu")   # 동적 양자화 모델은 CPU 추론 권장
MAX_LEN = 256

LABEL2ID = {'uncertain': 0, 'negative': 1, 'positive': 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

MODEL_PATH = "./model_emotion_quantized.pt"

class BertClassifier(nn.Module):
    def __init__(self, bert, hidden_size=768, num_classes=3, dr_rate=0.3, class_weights=None):
        super().__init__()
        self.bert = bert
        self.dropout = nn.Dropout(p=dr_rate) if dr_rate and dr_rate > 0 else nn.Identity()
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights) if class_weights is not None else nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled = out.pooler_output if getattr(out, "pooler_output", None) is not None else out[0][:, 0]
        logits = self.classifier(self.dropout(pooled))
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            return logits, loss
        return logits, None

# 1) 문장 분할
def split_sentences(paragraph: str) : 
    lines = [s.strip() for s in paragraph.strip().splitlines() if s.strip()]
    sents = []
    for line in lines : 
        pieces = re.split(r'(?<=[\.!?])\s+|(?<=다\.)\s+|(?<=요\.)\s+', line)
        sents += [p.strip() for p in pieces if p and p.strip()]
    return sents

# 2) 모델/ 토크나이저 로드
print("[Load] tokenizer =>")
tokenizer = get_tokenizer()

print("f[Load] quantized model => {MODEL_PATH}")
try :
    import sys as _sys
    main_mod = _sys.modules.get('__main__')
    if main_mod is not None and not hasattr(main_mod, 'BertClassifier'):
        setattr(main_mod, 'BertClassifier', BertClassifier)
except Exception as e:
    pass

try : 
    model = torch.load(MODEL_PATH, map_location=DEVICE)
except AttributeError : 
    import sys as _sys
    main_mod = _sys.modules.get('__main__')
    if main_mod is not None and hasattr(main_mod, 'BertClassifier'):
        setattr(main_mod, 'BertClassifier', BertClassifier)
    model = torch.load(MODEL_PATH, map_location=DEVICE)
model.eval()

# 3) 감정예측
@torch.no_grad() # 
def predict_sentences(sent_list) : 
    results, probs_all = [], [] # 리스트 형태로 받으면 문장단위 softmax확률을 probs_all에 저장
    for sent in sent_list : 
        enc = tokenizer(
            sent,
            padding = 'max_length',
            truncation = True,
            max_length = MAX_LEN,
            return_tensors = 'pt',
            return_token_type_ids = True
        )

        input_ids =  enc["input_ids"].to(DEVICE)
        attention_mask = enc["attention_mask"].to(DEVICE)
        token_type_ids = enc.get("token_type_ids", attention_mask.new_zeros(attention_mask.size())).to(DEVICE)

        out = model(input_ids, attention_mask, token_type_ids)
        logits = out[0] if isinstance(out, (tuple, list)) else out

        prob = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        pred_id = int(prob.argmax())
        results.append({
            "text": sent,
            "pred_id": pred_id,
            "pred_label": ID2LABEL[pred_id],
            "probs": {
                "uncertain": float(prob[LABEL2ID['uncertain']]),
                "negative":  float(prob[LABEL2ID['negative']]),
                "positive":  float(prob[LABEL2ID['positive']]),
            }
        })
        probs_all.append(prob)

    para_summary = None
    if probs_all:
        mean_prob = np.stack(probs_all, axis=0).mean(axis=0)
        para_summary = {
            "paragraph_pred_id": int(mean_prob.argmax()),
            "paragraph_pred_label": ID2LABEL[int(mean_prob.argmax())],
            "paragraph_probs": {
                "uncertain": float(mean_prob[LABEL2ID['uncertain']]),
                "negative":  float(mean_prob[LABEL2ID['negative']]),
                "positive":  float(mean_prob[LABEL2ID['positive']]),
            }
        }
    return results, para_summary

# 4) 테스트
if __name__ == "__main__" : 
    paragraph = """
    일본어 역량을 쌓고, 일본에서 생활하며 일본 문화와 일하는 방식에 대해 얻은 이해와 통찰은, 사람 관계와 매뉴얼이라는 두 가지 키워드를 들 수 있습니다.
먼저 사람 관계에 대해서는 한국에 비해 좀 더 개인 영역을 중시한다고 느꼈습니다. 회사에 입사하였을 때, 그리고 부서를 옮기게 되었을 때 사람에 익숙해지기 보다는 업무에 익숙해지는 것이 우선이었고, 업무 외의 개인 취미 등을 선뜻 함께하기 어려운 분위기가 있었습니다. 이는 지역 차이도 존재하여, 오사카에서는 비교적 이러한 부분이 적었지만, 도쿄에서는 같은 팀이면서도 제가 먼저 말을 걸기 전까지는 대화를 나누지 못한 경우도 있었습니다. 이러한 이해는 서로 다른 문화를 가진 팀원들이나 파트너 사와 업무를 진행하게 되었을 때 효과적으로 소통하는 데 활용될 것입니다.
다음으로 매뉴얼에 관해서는 일을 할 때 매뉴얼과 절차를 중시하는 방식이 중요하다고 느꼈습니다. 어떤 업무든 이를 따라야 한다는 점이 때로는 간단히 해결될 수 있는 문제도 오랜 시간과 노력을 들이게 만들곤 하였습니다. 하지만 결과적으로 실수를 줄이고, 설령 문제가 발생했을 때에도 이를 미리 예측하여 안정성을 높이는 효과도 존재했습니다. 일반적으로 일본에서 생활하거나 업무를 하는 사람들은 이를 답답하게 느끼곤 하지만, 이러한 통찰을 통해 업무 방식을 존중하고 따름으로써 프로젝트의 원활한 진행에 기여할 수 있을 것입니다.
"""
    sents = split_sentences(paragraph)
    sent_results, para_result = predict_sentences(sents)
    print("=== Sentence-level ===")
    for r in sent_results: 
        print(f"- {r['text']}")
        print(f" -> pred: {r['pred_label']} probs={r['probs']}")
        print("\n=== Paragraph-level (mean of probs) ===")
        print(para_result)