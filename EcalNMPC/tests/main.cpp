#include "nmpc/utils/files.hpp"
#include "nmpc/ControllerServer.h"

int main(const int argc, char** argv)
{
    eCAL::Initialize(argc, argv, "Control Server");

    const nmpc::Params nmpc_params = nmpc::readYaml("/home/vn/Documents/MyDrawft/CageStack_IBVS/EcalNMPC/configs/nmpc_st.ymal");
    std::cout << nmpc_params << std::endl;

    ControllerServer controller_server_st(nmpc_params);

    while (eCAL::Ok())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    eCAL::Finalize();
    // const auto reference = std::vector<double>{3.2, 777.6, 66.6, 0.1, 1.0};
    // const auto a = casadi::MX::eye(5) * reference;
    // casadi::Opti mNLP;
    // auto b = mNLP.parameter(5);
    // mNLP.set_value(b, reference);
    // const auto c = casadi::MX::eye(5) * b;
    // const auto d = casadi::MX::diag(b);
    // std::cout << a << std::endl;
    // std::cout << c << std::endl;
    // std::cout << d << std::endl;
    return 0;
}
