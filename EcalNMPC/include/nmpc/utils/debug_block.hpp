#pragma once
#include <utility>

#ifdef DEBUG_BLOCK
static constexpr bool gDebugBlock = true;
#else
static constexpr bool gDebugBlock = false;
#endif


template<class Func>
void debugBlock(Func&& block)
{
    if constexpr (gDebugBlock)
    {
        std::forward<Func>(block)();
    }
}
