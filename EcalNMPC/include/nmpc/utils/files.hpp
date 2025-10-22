#pragma once
#include "nmpc/kinematics.hpp"
#include <yaml-cpp/yaml.h>

namespace nmpc
{
inline Params readYaml(const std::string& file_path)
{
    static const auto getVar = [](const YAML::Node& yaml_node, const char* key, auto& var) -> void
        {
            var = yaml_node[key].as<std::decay_t<decltype(var)>>();
        };

    Params params{};

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

        getVar(root, "weight_Q", params.weight_Q);
        getVar(root, "weight_F", params.weight_F);
        getVar(root, "weight_R", params.weight_R);
    }
    catch (const YAML::BadFile& e)
    {
        std::cerr << "[YAML] Failed to open file: " << e.what() << std::endl;
        std::exit(EXIT_FAILURE);
    }
    catch (const YAML::Exception& e)
    {
        std::cerr << "[YAML] Failed to parse: " << e.what() << std::endl;
        std::exit(EXIT_FAILURE);
    }

    return params;
}
}

inline std::ostream& operator<<(std::ostream& os, const nmpc::Params& p)
{
    return os << "----------------------------- NMPC Params -----------------------------\n" <<
           "horizon: " << p.horizon << ", dt: " << p.dt << " s"
           << "\ninput_len: " << p.input_len << ", state_len: " << p.state_len << ", output_len: " << p.output_len
           << "\nwheel_base: " << p.wheel_base << " m, wheel_radius: " << p.wheel_radius << " m"
           << "\nmax_acc: " << p.max_acc << " rad/s^2, max_speed: " << p.max_speed << " rad/s"
           << "\nmax_steer_speed: " << p.max_steer_speed << " rad/s, max_steer_angle: " << p.max_steer_angle << " rad"
           << "\nweight_Q: " << p.weight_Q << "\nweight_F: " << p.weight_F << "\nweight_R: " << p.weight_R
           << "\n------------------------------------------------------------------------";
}
