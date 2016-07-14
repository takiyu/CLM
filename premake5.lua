-- premake5.lua
--
-- *** Please fix this script for your environment. ***
--
exec_sources = {
  "./src/main.cpp",
  "./src/train.cpp",
  "./src/yukiti.cpp",
  "./src/yukiti2.cpp",
}

workspace "RespRobotWorkspace"
  configurations { "release", "debug" }
  language "C++"
  basedir "build"
  buildoptions { "-std=c++03" }

  includedirs { "/usr/local/include" }
  libdirs { "/usr/local/lib" }
  libdirs { "./build/bin/release" }

  links { "opencv_core", "opencv_highgui", "opencv_objdetect",
          "opencv_videoio", "opencv_imgproc", "opencv_calib3d",
          "opencv_imgcodecs",  }  -- cv 3.1

  -- Configuration
  configuration "debug"
    defines { "DEBUG" }
    flags { "Symbols" }

  configuration "release"
    defines { "NDEBUG" }
    flags { "Optimize" }

-- My library
project "clm"
  kind "SharedLib"
  files { "./src/**.cpp", }
  removefiles { exec_sources }


-- Executable projects
for i, path in ipairs(exec_sources) do
  name = string.match(path, ".*/(.-)%.%w-$")  -- "dir/name.ext"
  if not name then
    name = string.match(path, "(.-)%.%w-$")  -- "name.ext"
  end

  project( name )
    kind "ConsoleApp"
    links { "clm" }
    files { path }
end
