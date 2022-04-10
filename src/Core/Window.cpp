#include "Window.h"
#include <GLObjects/Texture.h>
#include<opencv2/opencv.hpp>
#include <GLObjects/RenderContext.h>
#include <GLObjects/Framebuffer.h>
#include <memory>
#include <GLObjects/Shader.h>

cv::VideoCapture mVideo;
std::unique_ptr<gl::Texture2D> mTexture;
std::unique_ptr<gl::Framebuffer> mFramebuffer;
gl::Texture2D* mAttachedColor = nullptr;
std::unique_ptr<gl::Program> mFrameProgram;
std::unique_ptr<gl::Vertices> mQuad;

Window* thiz = nullptr;


const char* vertShader = GLSL(
	layout(location = 0) in vec2 a_Pos;
	layout(location = 1) in vec2 a_UV;

	out vec2 TexCoords;

	void main()
	{
		TexCoords = a_UV;
		gl_Position = vec4(a_Pos.x, a_Pos.y, 0.0, 1.0);
	}
);

//
const char* fragShader = GLSL(
	out vec4 FragColor;

	in vec2 TexCoords;

	uniform sampler2D screenTexture;


	float flare(vec2 U)                            // rotating hexagon 
	{
		vec2 A = sin(vec2(0, 1.57) + 0.0);
		U = abs(U * mat2(A, -A.y, A.x)) * mat2(2, 0, 1, 1.7);
		return .2 / max(U.x, U.y);                      // glowing-spiky approx of step(max,.2)
	  //return .2*pow(max(U.x,U.y), -2.);

	}


	vec2 r(vec2 x)
	{
		return fract(1e4 * sin((x) * 541.17));
	}

	vec4 r1(vec4 x)
	{
		return fract(1e4 * sin((x) * 541.17));
	}

	vec2 sr2(float x)
	{
		return r(vec2(x, x + .1)) * 2. - 1.;
	}

	vec4 sr3(float x)
	{
		return r1(vec4(x, x + .1, x + .2, 0)) * 2. - 1.;
	}

	vec4 stars(vec4 O, vec2 U)
	{
		vec2 iResolution = vec2(1280.0, 720.0);
		vec2 R = iResolution.xy;
		U = (U + U - R) / R.y;
		O -= O + .3;
		for (float i = 0.; i < 99.; i++)
			O += flare(U - sr2(i) * R / R.y)           // rotating flare at random location
			* r1(vec4(i + .2))                          // random scale
			* (1. + sin(r1(vec4(i + .3)) * 6.)) * .1  // time pulse
			* (1. + .1 * sr3(i + .4));               // random color - uncorrelated
			  //* (1.+.1*sr3(i));                  // random color - correlated
		return O;
	}

	void main()
	{
		vec3 col = texture(screenTexture, TexCoords).rgb;
		//FragColor = vec4(col, 1.0);

		/*float brightness = dot(col.rgb, vec3(0.2126, 0.7152, 0.0722));
		if (brightness > 0.6)
			FragColor = vec4(col.rgb, 1.0);
		else
		{
			FragColor = vec4(0.0, 0.0, 0.0, 1.0);
		}*/

		vec4 starColor = stars(vec4(1.0, 1.0, 1.0, 1.0), TexCoords);
		FragColor = vec4(1 - col, 1.0);
		//FragColor = starColor;
	}
);
//



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

	glfwSetKeyCallback(mGLFWWindow, Window::KeyCallback);
	glfwSetCursorPosCallback(mGLFWWindow, Window::MouseMoveCallback);
	glfwSetMouseButtonCallback(mGLFWWindow, Window::MouseButtonCallback);
	glfwSetScrollCallback(mGLFWWindow, Window::MouseScrollCallback);
	glfwSetFramebufferSizeCallback(mGLFWWindow, Window::FramebufferSizeCallback);

	thiz = this;
	gl::RenderContext::SetViewport(mWidth, mHeight);
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

unsigned char* cvMat2TexInput(cv::Mat& img)
{
	cvtColor(img, img, cv::ColorConversionCodes::COLOR_BGR2RGB);
	//flip(img, img, -1);
	return img.data;
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
	ImGui::StyleColorsClassic();

	// Setup Platform/Renderer backends
	ImGui_ImplGlfw_InitForOpenGL(mWindow->mGLFWWindow, true);
	ImGui_ImplOpenGL3_Init(glsl_version);


	const float fdim = 1.f;
	const int vert_count = 6;
	float quad_vert[] = {
		-fdim,  fdim, 0.f,  0.0f, 1.0f,
		-fdim, -fdim, 0.f,  0.0f, 0.0f,
		 fdim, -fdim, 0.f,  1.0f, 0.0f,

		-fdim,  fdim, 0.f,  0.0f, 1.0f,
		 fdim, -fdim, 0.f,  1.0f, 0.0f,
		 fdim,  fdim, 0.f,  1.0f, 1.0f
	};

	mQuad = std::make_unique<gl::Vertices>();
	mQuad->AddVBO(std::vector<gl::AttribType>({ gl::AttribType::POSITION, gl::AttribType::TEXTURE_UV }), vert_count, sizeof(quad_vert), quad_vert);

	mVideo.open("C:/Users/User/Desktop/video.mp4");

	if (!mVideo.isOpened())
	{
		fprintf(stderr, "Failed to load video.\n");
		exit(1);
	}

	mTexture = std::make_unique<gl::Texture2D>();

	mTexture->SetParameters({
		{gl::ParamName::WRAP_S, gl::ParamValue::CLAMP_TO_EDGE},
		{gl::ParamName::WRAP_T, gl::ParamValue::CLAMP_TO_EDGE},
		{gl::ParamName::MIN_FILTER, gl::ParamValue::LINEAR},
		{gl::ParamName::MAG_FILTER, gl::ParamValue::LINEAR},
		});

	cv::Mat frame;
	if (mVideo.read(frame))
	{
		int videoWidth = frame.cols;
		int videoHeight = frame.rows;
		unsigned char* image = cvMat2TexInput(frame);

		int type = frame.type();


		mTexture->SetTexture2D(0, gl::Format::RGB, videoWidth, videoHeight, 0, gl::Format::RGB, gl::DataType::UNSIGNED_BYTE, image);
	}

	gl::Shader<gl::ShaderType::VERTEX> vertSh;
	gl::Shader<gl::ShaderType::FRAGMENT> fragSh;

	int vertShSize = strlen(vertShader);
	int fragShSize = strlen(fragShader);

	vertSh.LoadSources(1, &vertShader, &vertShSize);
	fragSh.LoadSources(1, &fragShader, &fragShSize);

	mFrameProgram = std::make_unique<gl::Program>();

	mFrameProgram->Attach(&vertSh, &fragSh, NULL);

	mFrameProgram->Link();

	mFramebuffer = std::make_unique<gl::Framebuffer>();
	mFramebuffer->Init(mFrameProgram.get(), mTexture->GetWidth(), mTexture->GetHeight());
	mAttachedColor = mFramebuffer->AttachTexture(gl::AttachType::COLOR0, gl::Format::RGB, gl::Format::RGB, gl::DataType::UNSIGNED_BYTE,
		{ 
			{ gl::ParamName::WRAP_S, gl::ParamValue::CLAMP_TO_EDGE },
			{ gl::ParamName::WRAP_T, gl::ParamValue::CLAMP_TO_EDGE },
			{ gl::ParamName::MIN_FILTER, gl::ParamValue::LINEAR },
			{ gl::ParamName::MAG_FILTER, gl::ParamValue::LINEAR },
		}
	);

	if (!mFramebuffer->CheckFramebufferStatus())
	{
		assert("Failed frame buffer !");
	}

	/*srand(time(NULL));

	const int count = 1000;
	glm::vec2 uvs[count];
	for (int i = 0; i < count; i++)
	{
		glm::vec2 uv;
		uv.x = static_cast<float>(rand() % 100) / 100.f;
		uv.y = static_cast<float>(rand() % 100) / 100.f;
		uvs[i] = uv;
	}

	mFrameProgram->Use();
	for (unsigned int i = 0; i < 1000; i++)
	{
		int uvs_loc = mFrameProgram->Uniform(std::string("offsets[" + std::to_string(i) + "]").c_str());
		mFrameProgram->SetFloat2(uvs_loc, uvs[i]);
	}
	mFrameProgram->StopUsing();*/
	
	gl::RenderContext::SetClearColor(0.f, 0.4f, 0.3f, 1.f);
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
	//render in frame buffer
	mFramebuffer->Bind(gl::BindType::ReadAndDraw);
	gl::RenderContext::SetViewport(mAttachedColor->GetWidth(), mAttachedColor->GetHeight());
	gl::RenderContext::Clear(gl::BufferBit::COLOR);
	mFrameProgram->Use();
	int tex_loc = mFrameProgram->Uniform("screenTexture");
	mTexture->Activate(tex_loc);
	mQuad->Draw(gl::Primitive::TRIANGLES);
	mFrameProgram->StopUsing();
	mFramebuffer->UnBind(gl::BindType::ReadAndDraw);


	gl::RenderContext::Clear(gl::BufferBit::COLOR);
	cv::Mat frame;

	if (mVideo.read(frame))
	{
		int videoWidth = frame.cols;
		int videoHeight = frame.rows;
		unsigned char* image = cvMat2TexInput(frame);

		ImGui::Begin("Video", nullptr);

		float div = 2.f;

		ImGui::Image((ImTextureID)mTexture->GetId(), ImVec2(mTexture->GetWidth() / div, mTexture->GetHeight() / div));

		ImGui::Image((ImTextureID)mAttachedColor->GetId(), ImVec2(mAttachedColor->GetWidth() / div, mAttachedColor->GetHeight() / div));

		ImGui::End();

		mTexture->Update(0, 0, videoWidth, videoHeight, image);
	}

	static bool show_demo_window = true;
	ImGui::ShowDemoWindow(&show_demo_window);
}

void Window::KeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{

}
void Window::MouseMoveCallback(GLFWwindow* window, double xpos, double ypos)
{

}
void Window::MouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{

}
void Window::MouseScrollCallback(GLFWwindow* window, double xoffset, double yoffset)
{

}
void Window::FramebufferSizeCallback(GLFWwindow* window, int width, int height)
{
	if (thiz)
	{
		thiz->mWidth = width;
		thiz->mHeight = height;

	}
	gl::RenderContext::SetViewport(width, height);
	
}