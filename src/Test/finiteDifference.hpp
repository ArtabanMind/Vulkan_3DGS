#include "../common/GaussianTypes.hpp"
#include "../engine/VkEngine.hpp"
#include "../engine/VkBuffer.hpp"
#include "../engine/VkCompute.hpp"

void finiteDifferenceTest() {
    const float eps = 0.01f;
    const uint32_t testIdx = 0;  // G0 테스트
    float originalScale = gaussians[testIdx].scale.x;

    auto runForwardAndGetLoss = [&]() -> float {
        gs::uploadToBuffer(engine.device(), paramsBuffer, 
                           gaussians.data(), sizeof(GaussianParam) * GAUSS_COUNT);
        
        vkResetCommandBuffer(cmd, 0);
        VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        vkBeginCommandBuffer(cmd, &beginInfo);
        
        // forward만 실행 (gaussian.comp, loss.comp)
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, renderCtx.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                renderCtx.pipelineLayout, 0, 1, &renderCtx.descriptorSet, 0, nullptr);
        vkCmdPushConstants(cmd, renderCtx.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(RenderPC), &renderPC);
        vkCmdDispatch(cmd, (WIDTH + 15) / 16, (HEIGHT + 15) / 16, 1);
        
        // barrier
        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
        
        // loss
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, lossCtx.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                lossCtx.pipelineLayout, 0, 1, &lossCtx.descriptorSet, 0, nullptr);
        vkCmdPushConstants(cmd, lossCtx.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(LossPC), &lossPC);
        vkCmdDispatch(cmd, (WIDTH + 15) / 16, (HEIGHT + 15) / 16, 1);
        
        vkEndCommandBuffer(cmd);
        
        VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &cmd;
        vkQueueSubmit(engine.computeQueue(), 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(engine.computeQueue());
        
        float loss;
        gs::downloadFromBuffer(engine.device(), lossBuffer, &loss, sizeof(float));
        return loss;
    };

    // 1) Loss(scale + eps)
    gaussians[testIdx].scale.x = originalScale + eps;
    float Loss_plus = runForwardAndGetLoss();

    // 2) Loss(scale - eps)
    gaussians[testIdx].scale.x = originalScale - eps;
    float Loss_minus = runForwardAndGetLoss();

    // 3) 복원
    gaussians[testIdx].scale.x = originalScale;

    // 4) 수치적 gradient
    float numerical = (Loss_plus - Loss_minus) / (2.0f * eps);

    // 5) Analytic gradient (backward 실행해서 비교)
    // 기존 학습 루프 첫 iter에서 gradsInt[0].dScale.x 출력하면 됨

    printf("======== Finite Diff Test (G%d, scale.x) ========\n", testIdx);
    printf("  Loss(+eps): %.6f\n", Loss_plus);
    printf("  Loss(-eps): %.6f\n", Loss_minus);
    printf("  Numerical dL/dScale.x: %.6f\n", numerical);
    printf("================================================\n");
}