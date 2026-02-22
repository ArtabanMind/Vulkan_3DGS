## Phase 4-0 완료 ✅

### 완료 항목
- dL/dOpacity 수식 유도 + 구현
- dL/dScale 수식 유도 + 구현
- Forward/Backward **anisotropic 통일** (sx, sy 분리)
- 128×128 해상도, 1500 iter → 수렴 확인

### 핵심 수식 (anisotropic)
```glsl
// Forward (gaussian.comp, renderGaussiansCPU 둘 다)
float r2 = (diff.x * diff.x) / (sx * sx) + (diff.y * diff.y) / (sy * sy);
float gaussian = exp(-0.5 * r2);

// Backward - dL/dPosition
vec2 dGauss_dCenter = gaussian * vec2(diff.x / (sx * sx), diff.y / (sy * sy));

// Backward - dL/dOpacity (T 포함!)
float dL_dOpacity = dot(dL_dR, g.color) * gaussian * T;

// Backward - dL/dScale (T 포함!)
float dL_dG = dot(dL_dR, g.color) * g.opacity * T;
float dL_dsx = dL_dG * gaussian * dx * dx / (sx * sx * sx);
float dL_dsy = dL_dG * gaussian * dy * dy / (sy * sy * sy);
```

### 주요 삽질 & 교훈
| 문제 | 원인 | 해결 |
|------|------|------|
| scale 8.0 근처 수렴 | main.cpp에 `g.scale = 8.0f` 하드코딩 있었음 | 삭제 |
| gradient ≈ 0 | dL/dOpacity, dL/dScale에 T 누락 | `* T` 추가 |
| 검정 이미지 | scale 0.1f → 가우시안 점 하나 | scale 10.0f로 증가 |
| isotropic vs anisotropic 불일치 | forward/backward 수식 달랐음 | 둘 다 sx, sy 분리 |

### 현재 파라미터
- 해상도: 128×128
- 가우시안 수: 3
- Iteration: 1500
- LR: position 30.0f, color 0.3f, opacity 0.3f, scale 10.0f
- 타겟 scale: 10.0f

---

## 다음 Phase: 4-1 (Bitonic Sort, N=8)
- 목표 : Phase 4: 3D→2D 투영 ⬅️ 현재

4-1) Cov3D → Cov2D 변환 → Python/numpy 비교
4-2) MVP + 2D bounding box → 투영 좌표 검증
4-3) 투영 결과 시각적 확인