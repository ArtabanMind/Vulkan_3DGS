#pragma once
#include <glm/glm.hpp>
// ============================================================
// 파일: src/common/GaussianTypes.hpp (v2 - Phase 0 완성)
// 역할: 3DGS 핵심 구조체 정의 (SSBO와 1:1 매핑)
// ============================================================

namespace gs {

// ------------------------------------------------------------
// GaussianParam: 단일 가우시안의 학습 가능한 파라미터
// ------------------------------------------------------------
// INRIA 3DGS 기준 파라미터:
//   - position: 3D 위치
//   - scale: 3D 스케일 (공분산 행렬 구성에 사용)
//   - rotation: 회전 쿼터니언 (공분산 행렬 구성에 사용)
//   - opacity: 불투명도 (sigmoid 전 raw 값으로 저장할 수도 있음)
//   - color: SH degree 0 = RGB 직접 저장
//
// std430 메모리 레이아웃 (GPU 전송 시 중요):
//   vec3  → 16바이트 정렬 (12 + 4 padding)
//   float → 4바이트
//   vec4  → 16바이트 정렬
//
// 트릭: vec3 뒤에 float를 붙이면 padding 없이 16바이트 맞춤
// ------------------------------------------------------------
struct GaussianParam {
    // ---- 위치 + 불투명도 (16 bytes) ----
    glm::vec3 position;   // 3D 월드 좌표
                          // 예: (0, 0, 5) = 카메라 앞 5m
    float opacity;        // [0, 1], 1 = 완전 불투명
    
    // ---- 스케일 + padding (16 bytes) ----
    glm::vec3 scale;      // 각 축 방향 크기 (양수)
                          // 예: (0.1, 0.1, 0.1) = 반지름 ~0.1m 구형
                          // 예: (0.5, 0.1, 0.1) = x축으로 늘어난 타원체
    float _pad0;          // std430 정렬용 padding
    
    // ---- 회전 (16 bytes) ----
    glm::vec4 rotation;   // 쿼터니언 (w, x, y, z) - 정규화 필요
                          // 예: (1, 0, 0, 0) = 회전 없음 (identity)
                          // 예: (0.707, 0.707, 0, 0) = x축 90도 회전
    
    // ---- 색상 + padding (16 bytes) ----
    glm::vec3 color;      // RGB [0, 1], SH degree 0
                          // 예: (1, 0, 0) = 빨강
                          // Phase 8에서 SH coeffs 배열로 확장 예정
    float _pad1;          // std430 정렬용 padding
};

// 총 크기: 16 * 4 = 64 bytes
static_assert(sizeof(GaussianParam) == 64, 
    "GaussianParam must be 64 bytes for SSBO alignment");

// ------------------------------------------------------------
// 헬퍼: 기본값으로 초기화된 가우시안 생성
// ------------------------------------------------------------
inline GaussianParam makeDefaultGaussian(glm::vec3 pos, glm::vec3 col) {
    return GaussianParam{
        .position = pos,
        .opacity  = 1.0f,
        .scale    = glm::vec3(0.1f),        // 작은 구형
        ._pad0    = 0.0f,
        .rotation = glm::vec4(1, 0, 0, 0),  // identity quaternion
        .color    = col,
        ._pad1    = 0.0f
    };
}

}