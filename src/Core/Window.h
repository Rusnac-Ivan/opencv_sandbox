#ifndef _WINDOW_H_
#define _WINDOW_H_

#include "Platform.h"
#include "EventHandler.h"

#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>

#include <iostream>
#include <cstdint>

class Window
{
	class GUI
	{
	private:
		Window *mWindow;

	public:
		GUI(Window *window) : mWindow(window) {}
		~GUI() {}

		void Init(const char *glsl_version);
		void Render();

	private:
		void DrawElements();
	};

public:
	Window() : mWidth(0), mHeight(0), mGLFWWindow(nullptr), mGUI(this) {}
	~Window();

	void Create(uint32_t width, uint32_t height, const char *windowName);
	void PollEvents();

	void RenderGUI() { mGUI.Render(); }

	void SwapBuffers() { glfwSwapBuffers(mGLFWWindow); }
	bool WindowIsOpen() { return !glfwWindowShouldClose(mGLFWWindow); }

	uint32_t GetWidth() { return mWidth; }
	uint32_t GetHeight() { return mHeight; }

private:
	uint32_t mWidth;
	uint32_t mHeight;
	GLFWwindow *mGLFWWindow;
	GUI mGUI;

};

#endif