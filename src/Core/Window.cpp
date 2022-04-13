#include "Window.h"
#include <GLObjects/Texture.h>
#include<opencv2/opencv.hpp>
#include <GLObjects/RenderContext.h>
#include <GLObjects/Framebuffer.h>
#include <memory>
#include <GLObjects/Shader.h>
#include <opencv2/face.hpp>
#include <drawLandmarks.hpp>

cv::VideoCapture mVideo;
std::unique_ptr<gl::Texture2D> mTexture;
std::unique_ptr<gl::Framebuffer> mFramebuffer;
gl::Texture2D* mAttachedColor = nullptr;
std::unique_ptr<gl::Program> mFrameProgram;
std::unique_ptr<gl::Vertices> mQuad;
//opencv
cv::CascadeClassifier faceDetector;
cv::Ptr<cv::face::Facemark> facemark;
int u_Time_Loc = -1;;

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

//https://developer.nvidia.com/gpugems/gpugems/part-iii-materials/chapter-20-texture-bombing
const char* fragShader = GLSL(
	out vec4 FragColor;

	in vec2 TexCoords;

	uniform sampler2D screenTexture;
	uniform vec2 uResolution;
	uniform float uTime;

	mat2 Rot(float a)
	{
		float s = sin(a);
		float c = cos(a);
		return mat2(c, -s, s, c);
	}

	float Star(vec2 uv, float flare) {
		float d = length(uv);
		float m = .05 / d;

		float rays = max(0., 1. - abs(uv.x * uv.y * 1000.));
		m += rays * flare;
		uv *= Rot(3.1415 / 4.);
		rays = max(0., 1. - abs(uv.x * uv.y * 1000.));
		m += rays * .3 * flare;

		m *= smoothstep(0.5, .2, d);
		return m;
	}

	float Hash21(vec2 p) {
		p = fract(p * vec2(123.34, 456.21));
		p += dot(p, p + 45.32);
		return fract(p.x * p.y);
	}

	vec3 StarLayer(vec2 uv) {
		vec3 col = vec3(0);

		vec2 gv = fract(uv) - .5;
		vec2 id = floor(uv);

		for (int y = -1; y <= 1; y++) {
			for (int x = -1; x <= 1; x++) {
				vec2 offs = vec2(x, y);

				float n = Hash21(id + offs); // random between 0 and 1
				float size = 2.0;// fract(n * 345.32);

				float star = Star(gv - offs, smoothstep(2., 1., size) * .6);

				//vec3 color = sin(vec3(.2, .3, .9)*fract(n*2345.2)*123.2)*.5+.5;
				//color = color*vec3(1,.25,1.+size)+vec3(.2, .2, .1)*2.;
				vec3 color = vec3(1.0, 1.0, 1.0);

				star *= sin(uTime * 3. + n * 6.2831) * .5 + 1.;
				col += star * size * color;
			}
		}
		return col;
	}

	//const float scale = 10.0;
	const float NUM_LAYERS = 7.0;

	void main()
	{
		vec2 uv = (gl_FragCoord.xy - 0.5 * uResolution.xy) / uResolution.y;
		vec3 col = texture(screenTexture, TexCoords).rgb;

		vec3 res_color = vec3(0.);

		float brightness = dot(col.rgb, vec3(0.2126, 0.7152, 0.0722));
		if (brightness > 0.7)
		{
			//res_color = col.rgb;

			for (float i = 0.; i < 1.; i += 1. / NUM_LAYERS) {
				float depth = fract(i);

				float scale = mix(20., 10., depth);

				vec2 gv = fract(uv * scale) - .5;
				vec2 id = floor(uv * scale);
				float step = 1. / scale;
				vec2 cell_center_uv = id * step + vec2(step / 2.);

				//float brightness = dot(texture(screenTexture, cell_center_uv).rgb, vec3(0.2126, 0.7152, 0.0722));

				//if (brightness > 0.7)
				{
					float fade = depth * smoothstep(1., .4, depth);
					res_color += StarLayer(uv * scale + i * 453.2) * fade;
				}
			}
			
		}
		else
		{
			res_color = vec3(0.0, 0.0, 0.0);
		}

		//vec4 starColor = stars(vec4(1.0, 1.0, 1.0, 1.0), TexCoords);
		//FragColor = vec4(col, 1.0);
		//FragColor = starColor;

		FragColor = vec4(res_color, 1.0);

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

	mVideo.open("C:/Users/ion/Desktop/video1.mp4");

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

	int resolution_loc =  mFrameProgram->Uniform("uResolution");
	mFrameProgram->Use();
	mFrameProgram->SetFloat2(resolution_loc, glm::vec2(mTexture->GetWidth(), mTexture->GetHeight()));
	mFrameProgram->StopUsing();

	u_Time_Loc = mFrameProgram->Uniform("uTime");

	//opencv
	//-- 1. Load the cascades
	/*if (!faceDetector.load("D:/CPP/opencv_sandbox/build/installed/Windows/opencv/etc/haarcascades/haarcascade_frontalface_alt2.xml"))
	{
		assert("failed face_cascade_name !");
	};

	// Create an instance of Facemark
	facemark = cv::face::FacemarkLBF::create();

	// Load landmark detector
	facemark->loadModel("D:/CPP/opencv_sandbox/build/GSOC2017/src/GSOC2017/data/lbfmodel.yaml");

	// Variable to store a video frame and its grayscale */

	
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
	mFrameProgram->SetFloat(u_Time_Loc, (float)glfwGetTime());
	mQuad->Draw(gl::Primitive::TRIANGLES);
	mFrameProgram->StopUsing();
	mFramebuffer->UnBind(gl::BindType::ReadAndDraw);




	gl::RenderContext::Clear(gl::BufferBit::COLOR);
	cv::Mat frame;

	if (mVideo.read(frame))
	{
		/*cv::Mat gray;

		// Find face
		std::vector<cv::Rect> faces;
		// Convert frame to grayscale because
		// faceDetector requires grayscale image.
		cvtColor(frame, gray, cv::ColorConversionCodes::COLOR_BGR2GRAY);

		// Detect faces
		faceDetector.detectMultiScale(gray, faces);

		// Variable for landmarks. 
		// Landmarks for one face is a vector of points
		// There can be more than one face in the image. Hence, we 
		// use a vector of vector of points. 
		std::vector< std::vector<cv::Point2f> > landmarks;

		// Run landmark detector
		bool success = facemark->fit(frame, faces, landmarks);

		if (success)
		{
			// If successful, render the landmarks on the face
			for (int i = 0; i < landmarks.size(); i++)
			{
				drawLandmarks(frame, landmarks[i]);
			}
		}*/


		int videoWidth = frame.cols;
		int videoHeight = frame.rows;
		unsigned char* image = cvMat2TexInput(frame);

		ImGui::SetNextWindowPos(ImVec2(0.f, 0.f), ImGuiCond_Always);
		ImGui::SetNextWindowSize(ImVec2(mTexture->GetWidth() * 2.f, 800.f), ImGuiCond_Always);
		ImGui::Begin("Video", nullptr);
		{
			float div = 1.f;

			ImGui::Image((ImTextureID)mTexture->GetId(), ImVec2(mTexture->GetWidth() / div, mTexture->GetHeight() / div));
			ImGui::SameLine();
			ImGui::Image((ImTextureID)mAttachedColor->GetId(), ImVec2(mAttachedColor->GetWidth() / div, mAttachedColor->GetHeight() / div));
		}
		ImGui::End();

		mTexture->Update(0, 0, videoWidth, videoHeight, image);
	}

	//static bool show_demo_window = true;
	//ImGui::ShowDemoWindow(&show_demo_window);
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