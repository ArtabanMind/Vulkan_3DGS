좋은 질문들입니다, 마스터! 🐕

---

**Q1: std430 규칙**

| 타입 | 정렬 | 크기 |
|------|------|------|
| float | 4 | 4 |
| vec2 | 8 | 8 |
| vec3 | **16** | 12 (+ 4 padding) |
| vec4 | 16 | 16 |
| mat4 | 16 | 64 |

```
핵심: vec3 단독 = 16바이트 정렬 (4바이트 낭비)
트릭: vec3 + float 연속 배치 → padding 없이 16바이트
```

**std140 vs std430:**
- std140: UBO용, 더 엄격 (vec3 뒤 무조건 padding)
- std430: SSBO용, 더 효율적 (우리가 사용)

---

**Q2: local_size_z 필요성 + 최대 크기**

```glsl
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
```

| 차원 | 용도 예시 |
|------|-----------|
| x만 | 1D 배열 (가우시안 N개) ← 우리 경우 |
| x, y | 2D 이미지 (픽셀 처리) |
| x, y, z | 3D 볼륨 (의료 영상 등) |

**최대 크기 (RTX 1080 기준):**
```
local_size_x * y * z ≤ 1024 (총 스레드)
각 축 최대: x=1024, y=1024, z=64
```

**권장:** 64 또는 256 (warp 크기 32의 배수)

---

**Q3: Descriptor Type 종류 (두괄식)**

| Type | 읽기/쓰기 | 크기 | 용도 |
|------|-----------|------|------|
| **STORAGE_BUFFER (SSBO)** | R/W | 큼 (GB) | 가우시안 배열 ← 우리 |
| UNIFORM_BUFFER (UBO) | R only | 작음 (64KB) | 상수 (MVP 행렬) |
| STORAGE_IMAGE | R/W | 이미지 | 렌더 타겟 |
| SAMPLED_IMAGE | R only | 이미지 | 텍스처 |
| SAMPLER | - | - | 샘플링 설정 |

**학습 시 우리가 쓸 것:**
- SSBO: params, grads, 중간값
- STORAGE_IMAGE: 렌더 결과 (backward에서 읽음)
- UBO: 카메라, 학습률 등 상수

확장 필요하시면 말씀해주세요.

---

**Q4: descriptorCount = 4 이유**

```cpp
poolSize.descriptorCount = 4;
```

| 용도 | binding | 개수 |
|------|---------|------|
| params | 0 | 1 |
| grads | 1 | 1 (Phase 2) |
| target image | 2 | 1 (Phase 2) |
| 여유분 | - | 1 |

**지금:** 1개만 씀
**나중:** backward 추가 시 3~4개 필요

넉넉하게 잡아둔 것. 부족하면 pool 재생성해야 해서 귀찮음.

---

**Q5: pipelineLayoutInfo vs layoutInfo**

```
layoutInfo         → DescriptorSetLayout 생성
                     "이 shader는 binding 0에 SSBO 씀" (계약서)

pipelineLayoutInfo → PipelineLayout 생성
                     "이 pipeline은 위 layout을 사용함" (계약서 묶음)
```

**비유:**
```
DescriptorSetLayout = 주문서 양식 (어떤 항목이 필요한지)
PipelineLayout      = 주문서 양식 + 서명란 묶은 계약서
Pipeline            = 실제 체결된 계약
```

---

**Q6: 왜 Pool을 만드는가?**

```
Pool = 메모리 사전 할당 → 빠른 개별 할당
```

**비유:**
```
Pool 없이: 매번 은행 가서 계좌 개설 (느림)
Pool 있음: 미리 통장 10개 만들어둠 → 필요할 때 바로 지급 (빠름)
```

| Pool | 할당 대상 |
|------|-----------|
| CommandPool | CommandBuffer |
| DescriptorPool | DescriptorSet |

**Vulkan 철학:** 런타임 할당 최소화 → 성능

---

**Q7: allocateInfo에 pool + layout 둘 다 필요한 이유**

```cpp
allocInfo.descriptorPool     = ctx.descriptorPool;    // 메모리 어디서?
allocInfo.pSetLayouts        = &ctx.descriptorSetLayout;  // 구조가 뭐야?
```

```
Pool   = "메모리 창고" (어디서 할당?)
Layout = "설계도" (어떤 구조로?)

DescriptorSet = 설계도대로 창고에서 메모리 받아 만든 실제 객체
```

**비유:**
```
Pool   = 레고 블록 상자
Layout = 조립 설명서
Set    = 설명서대로 조립한 완성품
```

---

더 깊이 파고 싶은 항목 있으면 말씀해주세요! 아니면 다음 단계 (main.cpp 실행) 진행할까요?