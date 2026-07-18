# Introduction to Machine Learning — 예상 시험 & 모범답안
### UNLV · Prof. Mingon Kang · 영문 문제 + 한글 번역 + 모범답안

> **시험 형식**: 총 20문항 — True/False 10 · Short Answer 5 · Essay 3 · Problem Solving 2. 전부 영어, 계산기·노트 불가.
> 아래는 리뷰세션 강조 내용 + 교수님 연구 취향(불균형 데이터·평가지표·딥러닝)을 반영한 **예상 문제**와 모범답안입니다.

---

## Part 1. True / False (10문항)

**Q1. A single perceptron can solve the XOR problem.**
- (한글) 단일 퍼셉트론은 XOR 문제를 풀 수 있다.
- **정답: False (거짓)**
- **Why**: XOR is not linearly separable, and a single perceptron can only draw one linear boundary.
- (한글) XOR은 선형 분리가 불가능한데, 단일 퍼셉트론은 직선(초평면) 하나만 그을 수 있어 풀 수 없다.

**Q2. The number of nodes in the input layer can be chosen freely by the designer.**
- (한글) 입력층 노드 수는 설계자가 임의로 정할 수 있다.
- **정답: False (거짓)**
- **Why**: The number of input nodes must equal the number of input features.
- (한글) 입력 노드 수는 데이터의 feature(특징) 개수와 같아야 한다.

**Q3. Logistic regression has a closed-form solution like OLS.**
- (한글) 로지스틱 회귀는 OLS처럼 닫힌 형태(closed-form)의 해가 있다.
- **정답: False (거짓)**
- **Why**: Logistic regression has no closed form; it is solved iteratively via MLE / gradient descent.
- (한글) 로지스틱 회귀는 닫힌 해가 없고, 최대우도추정(MLE)을 gradient descent로 근사해 푼다.

**Q4. Increasing K in KNN makes the decision boundary smoother.**
- (한글) KNN에서 K를 크게 하면 결정 경계가 부드러워진다.
- **정답: True (참)**
- **Why**: A larger K averages over more neighbors, smoothing the boundary and reducing overfitting.
- (한글) K가 크면 더 많은 이웃을 반영해 경계가 부드러워지고 과적합이 줄어든다. (K가 작으면 과적합.)

**Q5. Accuracy is a reliable metric for highly imbalanced datasets.**
- (한글) 정확도(accuracy)는 불균형이 심한 데이터에서 신뢰할 만한 지표이다.
- **정답: False (거짓)**
- **Why**: With e.g. 9990:10 data, predicting all-majority gives 99.9% accuracy yet is useless. Use precision/recall/F1.
- (한글) 예로 9990:10 데이터에서 전부 다수클래스로 찍어도 정확도 99.9% → 무의미. precision/recall/F1을 써야 한다.

**Q6. Normalization parameters (mean/std) should be estimated using both training and test data.**
- (한글) 정규화 파라미터(평균/표준편차)는 훈련+테스트 데이터를 함께 써서 계산해야 한다.
- **정답: False (거짓)**
- **Why**: Fit on the training set only and apply to the test set; using test data causes data leakage.
- (한글) 훈련 데이터로만 계산해 테스트에 적용한다. 테스트를 쓰면 정보 누수(data leakage)다.

**Q7. The gradient ∇f points in the direction of steepest descent.**
- (한글) 그래디언트 ∇f는 가장 가파르게 감소하는 방향을 가리킨다.
- **정답: False (거짓)**
- **Why**: ∇f points in the direction of steepest ascent; gradient descent moves in the negative gradient direction.
- (한글) ∇f는 가장 가파르게 **증가**하는 방향이다. 경사하강법은 그 **반대(음의)** 방향으로 이동한다.

**Q8. AlexNet was the first convolutional neural network ever proposed.**
- (한글) AlexNet은 최초로 제안된 CNN이다.
- **정답: False (거짓)**
- **Why**: LeNet (LeCun, 1990s) came first; AlexNet (2012) popularized CNNs.
- (한글) 최초는 LeNet(LeCun, 1990년대). AlexNet(2012)은 CNN을 대중화시킨 사례다.

**Q9. In Ridge regression, (X'X + λI) is always invertible.**
- (한글) Ridge 회귀에서 (X'X + λI)는 항상 역행렬이 존재한다.
- **정답: True (참)**
- **Why**: Adding λI (λ>0) makes the matrix positive-definite, so Ridge always has a unique solution even under multicollinearity.
- (한글) λI(λ>0)를 더하면 양의 정부호가 되어 항상 역행렬 존재 → 다중공선성에도 유일해를 가진다.

**Q10. In K-fold cross-validation, the held-out fold is used to train the model.**
- (한글) K-겹 교차검증에서 남겨둔 fold는 모델 학습에 사용된다.
- **정답: False (거짓)**
- **Why**: The held-out fold is used for testing; the other K−1 folds are used for training.
- (한글) 남긴 fold는 테스트용이고, 나머지 K−1개로 학습한다.

---

## Part 2. Short Answer (5문항)

**Q11. Explain the difference between multi-class and multi-label classification.**
- (한글) 다중클래스 분류와 다중라벨 분류의 차이를 설명하라.
- **Answer**: In multi-class classification each sample has exactly **one** label chosen from three or more classes. In multi-label classification each sample can have **several** labels at the same time.
- (한글) 다중클래스는 샘플당 라벨이 **정확히 1개**(클래스는 3개 이상). 다중라벨은 샘플 하나가 **여러 라벨**을 동시에 가질 수 있다.

**Q12. How is the number of output-layer nodes decided for binary, multi-class, and regression problems?**
- (한글) 이진분류·다중클래스·회귀에서 출력층 노드 수는 어떻게 정하는가?
- **Answer**: Binary → 1 node (sigmoid) or 2 (softmax). Multi-class → one node per class (softmax). Regression → 1 node (linear, no activation). The output layer is determined by *what* you predict.
- (한글) 이진분류→1개(sigmoid) 또는 2개(softmax), 다중클래스→클래스 개수만큼(softmax), 회귀→1개(활성화 없이 linear). 출력층은 "무엇을 예측하느냐"로 결정된다.

**Q13. State one cause and one solution of the vanishing gradient problem.**
- (한글) 기울기 소실(vanishing gradient) 문제의 원인 하나와 해결책 하나를 쓰라.
- **Answer**: Cause — in deep networks, gradients shrink exponentially as they pass through many layers (especially with sigmoid, whose derivative is small, multiplied via the chain rule). Solution — use ReLU activation and/or Batch Normalization.
- (한글) 원인: 깊은 층을 거치며 chain rule로 작은 값이 계속 곱해져 기울기가 지수적으로 작아짐(특히 sigmoid). 해결: ReLU 활성화, Batch Normalization 등.

**Q14. Why is standardization important for a distance-based model such as KNN?**
- (한글) KNN 같은 거리 기반 모델에서 표준화(standardization)가 왜 중요한가?
- **Answer**: KNN decisions depend on distances, so features with larger numeric scales dominate the distance. Standardization puts all features on the same scale so each contributes fairly.
- (한글) KNN은 거리로 판단하는데, 스케일이 큰 feature가 거리를 지배해버린다. 표준화로 모든 feature를 같은 범위로 맞춰 공정하게 반영되게 한다.

**Q15. What is the role of the parameter C in soft-margin SVM?**
- (한글) 소프트마진 SVM에서 파라미터 C의 역할은?
- **Answer**: C controls the trade-off between margin width and misclassification. Large C penalizes errors heavily (narrow margin, strict); small C tolerates more errors (wider margin).
- (한글) C는 margin 폭과 오분류 사이의 트레이드오프를 조절한다. C가 크면 오류에 큰 페널티(margin 좁고 엄격), 작으면 오류 허용(margin 넓음).

---

## Part 3. Essay (3문항)

**Q16. Describe the backpropagation algorithm in its five steps (including the forward pass), and explain why backpropagation is needed.**
- (한글) 역전파 알고리즘의 5단계를 순전파 포함해 설명하고, 역전파가 왜 필요한지 설명하라. *(최우선 출제)*

**Answer (English):**
1. **Input** — set the input-layer activation a¹ = x.
2. **Feedforward** — for each layer l, compute the weighted sum z^l = W^l a^(l−1) + b^l and the activation a^l = f(z^l), up to the output layer, then compute the cost C from the output a^L and target y.
3. **Output error** — δ^L = ∂C/∂a^L ⊙ f'(z^L).
4. **Backpropagate the error** — for l = L−1, …, 2: δ^l = ((W^(l+1))^T δ^(l+1)) ⊙ f'(z^l).
5. **Update** — gradients are ∂C/∂W^l = δ^l (a^(l−1))^T and ∂C/∂b^l = δ^l; update parameters by gradient descent.
**Why needed**: A neural network has a very large number of parameters, so computing each gradient independently is extremely expensive. Backpropagation uses the **chain rule** together with **dynamic programming** (reusing the intermediate errors δ) to compute the gradients of *all* parameters in a single backward pass.

**모범답안 (한글):**
1. **입력**: 입력층 활성값 a¹ = x로 설정.
2. **순전파**: 각 층에서 가중합 z^l = W^l a^(l−1) + b^l과 활성화 a^l = f(z^l)을 출력층까지 계산하고, 출력 a^L과 정답 y로 비용 C 계산.
3. **출력층 오차**: δ^L = ∂C/∂a^L ⊙ f'(z^L).
4. **오차 역전파**: l = L−1, …, 2에 대해 δ^l = ((W^(l+1))^T δ^(l+1)) ⊙ f'(z^l).
5. **가중치·편향 갱신**: ∂C/∂W^l = δ^l (a^(l−1))^T, ∂C/∂b^l = δ^l 를 이용해 gradient descent로 갱신.
**왜 필요한가**: 신경망은 파라미터가 매우 많아 각 gradient를 하나씩 직접 계산하면 비효율적이다. 역전파는 **연쇄법칙(chain rule)** + **동적계획법**(중간 오차 δ 재사용)으로 **모든 파라미터의 gradient를 한 번의 역방향 계산**으로 효율적으로 구한다.

---

**Q17. Explain why a single perceptron is insufficient and why a multi-layer perceptron (MLP) is needed, using XOR as an example.**
- (한글) 단일 퍼셉트론이 왜 부족하고 왜 다층 퍼셉트론(MLP)이 필요한지 XOR 예시로 설명하라.

**Answer (English):** A single perceptron computes one linear boundary (wᵀx + b), so it can only classify **linearly separable** data. XOR — (0,0)→0, (1,1)→0, (0,1)→1, (1,0)→1 — cannot be separated by any single straight line, so a single perceptron fails. Adding a **hidden layer** lets the network combine several linear boundaries into a **nonlinear** decision region, so an MLP can represent XOR. By the **Universal Approximation Theorem**, an MLP with even one hidden layer can approximate any continuous function. This 1969 limitation caused an "AI winter" until MLP + backpropagation revived neural networks.

**모범답안 (한글):** 단일 퍼셉트론은 선형 경계(wᵀx + b) 하나만 만들어 **선형 분리 가능한** 데이터만 분류할 수 있다. XOR — (0,0)→0, (1,1)→0, (0,1)→1, (1,0)→1 — 은 어떤 직선으로도 분리할 수 없어 단일 퍼셉트론으로 못 푼다. **은닉층**을 추가하면 여러 선형 경계를 조합해 **비선형** 결정 영역을 만들 수 있어 MLP는 XOR을 표현할 수 있다. **보편근사정리**에 따르면 은닉층이 하나만 있어도 임의의 연속함수를 근사할 수 있다. 이 한계(1969)가 AI 겨울을 불렀고, MLP + 역전파로 신경망이 부활했다.

---

**Q18. Why do we use a train-test split and K-fold cross-validation? Explain how using test data during preprocessing causes data leakage.**
- (한글) 왜 train-test 분할과 K-겹 교차검증을 하는가? 전처리에 테스트 데이터를 쓰면 왜 data leakage가 되는지 설명하라.

**Answer (English):** A model evaluated on its own training data gives an over-optimistic score because it may simply memorize the data; this does not reflect **generalization** to new data. The test set — unseen during training — gives an unbiased estimate of real-world performance. **K-fold CV** splits the data into K folds, trains on K−1 and tests on the remaining fold, rotating K times and averaging the results; this reduces the variance caused by a single "unlucky" split and uses limited data efficiently. **Data leakage** happens when test information influences training — for example, computing normalization statistics (mean/std) over the *entire* dataset lets test information leak into training, producing falsely optimistic results. Preprocessing parameters must be estimated on the **training set only** and then applied to the test set.

**모범답안 (한글):** 모델을 자신의 훈련 데이터로 평가하면 데이터를 외웠을 수 있어 성능이 과대평가되며, 새 데이터에 대한 **일반화** 성능을 반영하지 못한다. 학습에 안 쓴 **테스트셋**은 실제 성능을 편향 없이 추정하게 해준다. **K-겹 CV**는 데이터를 K개로 나눠 K−1개로 학습·1개로 테스트를 K번 반복해 평균 내며, 한 번의 "운 나쁜 분할"에 따른 변동을 줄이고 한정된 데이터를 효율적으로 쓴다. **Data leakage**는 테스트 정보가 학습에 영향을 줄 때 생긴다 — 예: 정규화 통계(평균/표준편차)를 **전체** 데이터로 계산하면 테스트 정보가 학습에 새어 들어가 성능이 거짓으로 좋아 보인다. 전처리 파라미터는 **훈련셋에서만** 구해 테스트에 적용해야 한다.

---

## Part 4. Problem Solving (2문항)

**Q19. Compute the Euclidean distance between a = (2, 4, 1) and b = (5, 8, 2). Show all steps.**
- (한글) 두 점 a = (2, 4, 1), b = (5, 8, 2) 사이의 유클리드 거리를 구하라. 풀이 과정을 보이라. *(출제 확정)*

**Solution:**
```
dist(a,b) = sqrt( (a1−b1)² + (a2−b2)² + (a3−b3)² )
          = sqrt( (2−5)²   + (4−8)²   + (1−2)²   )
          = sqrt( (−3)²    + (−4)²    + (−1)²    )
          = sqrt( 9 + 16 + 1 )
          = sqrt(26)
          ≈ 5.10
```
- (한글 주의) 각 차원 차이를 먼저 제곱해 더한 뒤 **마지막에 한 번만** 제곱근(sqrt). √26 또는 ≈5.10 둘 다 정답으로 인정되는 경우가 많으나, 루트 안 정수(26)까지는 반드시 정확히.

**Q20. Given a confusion matrix with TP = 40, FP = 10, FN = 20, TN = 30, compute Precision, Recall, F1-score, and Accuracy. Explain why accuracy alone can be misleading.**
- (한글) 혼동행렬 TP=40, FP=10, FN=20, TN=30일 때 Precision, Recall, F1, Accuracy를 구하고, 정확도만으로는 왜 오해를 부를 수 있는지 설명하라.

**Solution:**
```
Precision = TP / (TP + FP) = 40 / (40 + 10) = 40/50 = 0.80
Recall    = TP / (TP + FN) = 40 / (40 + 20) = 40/60 ≈ 0.667
F1        = 2·P·R / (P + R) = 2·0.80·0.667 / (0.80 + 0.667)
          = 1.067 / 1.467 ≈ 0.727
Accuracy  = (TP + TN) / (TP+TN+FP+FN) = (40 + 30) / 100 = 0.70
```
**Why accuracy can mislead**: Accuracy weights every class equally, so on imbalanced data a high accuracy can hide poor performance on the minority (positive) class. Precision, recall, and F1 focus on the positive class and reveal that trade-off.
- (한글) 정확도는 모든 클래스를 동등하게 반영하므로, 불균형 데이터에서는 높은 정확도가 소수(양성) 클래스의 낮은 성능을 감출 수 있다. Precision/Recall/F1은 양성 클래스에 초점을 맞춰 그 한계를 드러낸다.

---

## 시험 직전 핵심 체크 (우선순위)
1. **Backprop 5단계 + forward pass**를 영어로 말할 수 있게 (Essay 최우선)
2. **Euclidean distance 손계산** 3세트 이상 연습 (확정)
3. **Confusion matrix 지표 6개 공식** 암기 + 숫자 계산
4. **KNN pseudocode / SVM margin 개념 / 단일 퍼셉트론 한계(XOR)**
5. **T/F 함정 10개**(Part 1) 통째로 훑기
