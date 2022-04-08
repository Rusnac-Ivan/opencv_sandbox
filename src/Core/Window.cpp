#include "Window.h"

static void glfw_error_callback(int error, const char* description)
{
	fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

void GLAPIENTRY 
MessageCallback(GLenum source,
	GLenum type,
	GLuint id,
	GLenum severity,
	GLsizei length,
	const GLchar* message,
	const void* userParam)
{
	fprintf(stderr, "OpenGL CALLBACK: %-s\n\ttype = 0x%x\n\tseverity = 0x%x\n\tmessage = %s\n\n",
		(type == GL_DEBUG_TYPE_ERROR ? "!!! GL ERROR !!!" : ""),
		type, severity, message);
	assert(type != GL_DEBUG_TYPE_ERROR && "OpenGL throw ERROR!");
}

void Window::Create(uint32_t width, uint32_t height, const char* windowName)
{
	glfwSetErrorCallback(glfw_error_callback);

	mWidth = width;
	mHeight = height;

	if (!glfwInit())
	{
		fprintf(stderr, "Failed to initialize GLFW library!\n");
		exit(1);
	}


	const char* glsl_version = "#version 460";
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_SAMPLES, 8);
	glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE);

	mGLFWWindow = glfwCreateWindow(mWidth, mHeight, windowName, nullptr, nullptr);

	if (mGLFWWindow == nullptr)
	{
		fprintf(stderr, "Failed to create GLFW window!\n");
		exit(1);
	}


	glfwMakeContextCurrent(mGLFWWindow);
	glfwSwapInterval(1);

	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		fprintf(stderr, "Failed to initialize GLAD!\n");
		exit(1);
	}

	/*if (gladLoadGL() == NULL)
	{
		fprintf(stderr, "Failed to initialize OpenGL loader!\n");
		exit(1);
	}*/

	glEnable(GL_DEBUG_OUTPUT);
	glDebugMessageCallback(MessageCallback, NULL);

	mGUI.Init(glsl_version);

	if (glfwRawMouseMotionSupported())
		glfwSetInputMode(mGLFWWindow, GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);

	glfwSetKeyCallback(mGLFWWindow, EventHandler::KeyCallback);
	glfwSetCursorPosCallback(mGLFWWindow, EventHandler::MouseMoveCallback);
	glfwSetMouseButtonCallback(mGLFWWindow, EventHandler::MouseButtonCallback);
	glfwSetScrollCallback(mGLFWWindow, EventHandler::MouseScrollCallback);
	glfwSetFramebufferSizeCallback(mGLFWWindow, EventHandler::FramebufferSizeCallback);
}

void Window::PollEvents()
{
	glfwPollEvents();
	glfwGetFramebufferSize(mGLFWWindow, reinterpret_cast<int*>(&mWidth), reinterpret_cast<int*>(&mHeight));
}

Window::~Window()
{
	if (mGLFWWindow != nullptr)
	{
		glfwDestroyWindow(mGLFWWindow);
		glfwTerminate();
	}
}

void Window::GUI::Init(const char* glsl_version)
{
	// Setup Dear ImGui context
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO();

	io.WantCaptureMouse = false;
	io.WantCaptureKeyboard = false;

	// Setup Dear ImGui style
	ImGui::StyleColorsDark();

	// Setup Platform/Renderer backends
	ImGui_ImplGlfw_InitForOpenGL(mWindow->mGLFWWindow, true);
	ImGui_ImplOpenGL3_Init(glsl_version);
}

void Window::GUI::Render()
{
	// Start the Dear ImGui frame
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();

	DrawElements();

	// Rendering
	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void Window::GUI::DrawElements()
{
	static bool show_demo_window = true;
	ImGui::ShowDemoWindow(&show_demo_window);
}