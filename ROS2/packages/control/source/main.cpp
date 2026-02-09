// #include "control/node/ControlCmdPublisher.h"
//
// int main(const int argc, char** argv)
// {
//     rclcpp::init(argc, argv);
//     const auto control_cmd_pub = std::make_shared<ControlCmdPublisher>();
//     rclcpp::spin(control_cmd_pub);
//     rclcpp::shutdown();
//     return 0;
// }


#include <iostream>
#include <string>
#include <unordered_map>
#include <cctype>

std::unordered_map<std::string, int> parseSemanticLabels(const std::string& msg)
{
    std::unordered_map<std::string, int> out;
    const size_t n = msg.size();
    // find first '{'
    const size_t pos = msg.find('{');
    if (pos == std::string::npos)
        return out;
    size_t i = pos + 1;

    while (i < n)
    {
        // find next quote for the id token (either " or ')
        size_t q1 = msg.find_first_of("\"'", i);
        if (q1 == std::string::npos)
            break;
        char quote = msg[q1];

        // find closing quote for id
        size_t q2 = msg.find(quote, q1 + 1);
        if (q2 == std::string::npos)
            break;
        std::string idStr = msg.substr(q1 + 1, q2 - q1 - 1);

        // try parse integer id
        int id = 0;
        try { id = std::stoi(idStr); }
        catch (...)
        {
            i = q2 + 1;
            continue;
        }

        // find `"class"` (allow single or double quoted)
        size_t classKey = msg.find("\"class\"", q2);
        if (classKey == std::string::npos)
            classKey = msg.find("'class'", q2);
        if (classKey == std::string::npos)
            break;

        // find colon after "class"
        size_t colon = msg.find(':', classKey);
        if (colon == std::string::npos)
            break;

        // find start quote of the class value
        size_t vq1 = msg.find_first_of("\"'", colon + 1);
        if (vq1 == std::string::npos)
            break;
        char vquote = msg[vq1];

        // find end quote of the class value
        size_t vq2 = msg.find(vquote, vq1 + 1);
        if (vq2 == std::string::npos)
            break;
        std::string cls = msg.substr(vq1 + 1, vq2 - vq1 - 1);

        // store mapping: class name -> id
        out[cls] = id;

        // advance i past this object
        i = vq2 + 1;
    }

    return out;
}

// simple demo
int main()
{
    std::string data = R"({'0':{"class":"BACKGROUND"},"1":{"class":"UNLABELLED"},"2":{"class":"trailer"},"3":{"class":"pallet"}})";
    auto map = parseSemanticLabels(data);
    for (auto& p : map)
    {
        std::cout << "class=\"" << p.first << "\" -> id=" << p.second << '\n';
    }
    return 0;
}
