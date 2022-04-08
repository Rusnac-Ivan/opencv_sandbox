#ifndef _GUI_H_
#define _GUI_H_


#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>



class GUI
{
private:
    void DrawGUIElements()
    {
        static bool show_demo_window = true;
        ImGui::ShowDemoWindow(&show_demo_window);
    }
public:
	GUI() {}
	~GUI() 
    {
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
    }

	void Init(GLFWwindow* window, const char* glsl_version)
	{
        // Setup Dear ImGui context
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO(); (void)io;

        // Setup Dear ImGui style
        ImGui::StyleColorsDark();

        // Setup Platform/Renderer backends
        ImGui_ImplGlfw_InitForOpenGL(window, true);
        ImGui_ImplOpenGL3_Init(glsl_version);


	}

    void RenderGUIElements()
    {
        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        DrawGUIElements();

        // Rendering
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    }




private:


};

#endif