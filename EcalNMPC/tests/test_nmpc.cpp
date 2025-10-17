#include "colib/ControllerServer.h"
#include <yaml-cpp/yaml.h>

nmpc::Params readYaml(const std::string& file_path)
{
    static const auto getVar = [](const YAML::Node& yaml_node, const char* key, auto& var) -> void
        {
            var = yaml_node[key].as<std::decay_t<decltype(var)>>();
        };

    nmpc::Params params{};

    try
    {
        const YAML::Node root = YAML::LoadFile(file_path);

        getVar(root, "horizon", params.horizon);
        getVar(root, "dt", params.dt);

        getVar(root, "input_len", params.input_len);
        getVar(root, "state_len", params.state_len);
        getVar(root, "output_len", params.output_len);

        getVar(root, "wheel_base", params.wheel_base);
        getVar(root, "wheel_radius", params.wheel_radius);

        getVar(root, "max_acc", params.max_acc);
        getVar(root, "max_speed", params.max_speed);
        getVar(root, "max_steer_speed", params.max_steer_speed);
        getVar(root, "max_steer_angle", params.max_steer_angle);
    }
    catch (const YAML::BadFile& e)
    {
        std::cerr << "[YAML] Failed to open file: " << e.what() << "\n";
        std::exit(EXIT_FAILURE);
    }
    catch (const YAML::Exception& e)
    {
        std::cerr << "[YAML] Failed to parse: " << e.what() << "\n";
        std::exit(EXIT_FAILURE);
    }

    return params;
}

inline std::ostream& operator<<(std::ostream& os, const nmpc::Params& p)
{
    return os << "---------------------- NMPC Params ----------------------\n" <<
           "horizon: " << p.horizon << ", dt: " << p.dt
           << "\ninput_len: " << p.input_len << ", state_len: " << p.state_len << ", output_len: " << p.output_len
           << "\nwheel_base: " << p.wheel_base << ", wheel_radius: " << p.wheel_radius
           << "\nmax_acc: " << p.max_acc << "\nmax_speed: " << p.max_speed
           << "\nmax_steer_speed: " << p.max_steer_speed << ", max_steer_angle: " << p.max_steer_angle
           << "\n--------------------------------------------------------";
}

int main(const int argc, char** argv)
{
    eCAL::Initialize(argc, argv, "Test NMPC", eCAL::Init::All | eCAL::Init::TimeSync);

    std::cout << "master=" << eCAL::Time::IsMaster()
            << " synced=" << eCAL::Time::IsSynchronized() << "\n";

    const nmpc::Params nmpc_params = readYaml("/home/vn/Documents/MyDrawft/CageStack_IBVS/EcalNMPC/configs/nmpc.ymal");
    std::cout << nmpc_params << std::endl;

    ControllerServer nmpc_server(nmpc_params);

    while (eCAL::Ok())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    eCAL::Finalize();

    return 0;
}
