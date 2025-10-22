#include "nmpc/utils/files.hpp"
#include "nmpc/ControllerServer.h"

int main(const int argc, char** argv)
{
    eCAL::Initialize(argc, argv, "ST control server");

    const nmpc::Params nmpc_params = nmpc::readYaml(PARAM_PATH);
    std::cout << nmpc_params << std::endl;
    ControllerServer controller_server_st(nmpc_params);

    while (eCAL::Ok())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    eCAL::Finalize();

    return 0;
}
