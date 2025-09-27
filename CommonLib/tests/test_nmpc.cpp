#include "../include/colib/NMPC.h"

int main()
{
    NMPC nmpc;

    nmpc.setGoal({10.0, 2.0, 0.6});

    const auto begin = std::chrono::system_clock::now();
    const auto [u, x] = nmpc.solve2();
    const auto end = std::chrono::system_clock::now();

    const auto elapse = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
    std::cout << "Elapse: " << elapse << " ms." << std::endl;

    std::cout << "Control policy: " << std::endl;
    std::cout << u << std::endl;
    std::cout << "--------------------------------" << std::endl;
    std::cout << "State sequences: " << std::endl;
    std::cout << x << std::endl;
    return 0;
}

// #include <iostream>
//
// int main()
// {
//     std::cout << "Hello!" << std::endl;
//     return 0;
// }
