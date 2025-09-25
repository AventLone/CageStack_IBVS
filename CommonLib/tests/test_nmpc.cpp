#include "../include/colib/NMPC.h"

int main()
{
    NMPC nmpc;

    nmpc.setGoalAndState({3.0, 4.0, M_PI / 3.4});

    auto begin = std::chrono::system_clock::now();
    const auto [fst, snd] = nmpc.solve();
    auto end = std::chrono::system_clock::now();

    auto elapse = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
    std::cout << "Elapse: " << elapse << std::endl;

    std::cout << fst.transpose() << std::endl;
    std::cout << snd.transpose() << std::endl;
    return 0;
}

// #include <iostream>
//
// int main()
// {
//     std::cout << "Hello!" << std::endl;
//     return 0;
// }
