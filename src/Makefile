MAIN_SRCS = $(wildcard *.cpp) $(wildcard test/*.cpp)

OBJ_DIR = ../obj
BIN_DIR = ../bin
NOMAKE_SRC= .git% data%
INCLUDES = 
OPTION = -I/usr/local/include/opencv -I/usr/local/include/opencv2 -L/usr/local/lib -lopencv_core -lopencv_highgui -lopencv_objdetect -lopencv_contrib -lopencv_imgproc -lopencv_ml -lopencv_gpu

GCC = g++
CFLAGS = -O2
D_OPTION = -MD

TARGET = $(addprefix $(BIN_DIR)/, $(MAIN_SRCS:.cpp=.out))
ALL_SRCS = $(shell find * | grep '.cpp$$')
LIB_SRCS = $(filter-out $(MAIN_SRCS) $(NOMAKE_SRC), $(ALL_SRCS))
ALL_OBJS = $(addprefix $(OBJ_DIR)/, $(ALL_SRCS:.cpp=.o))
LIB_OBJS = $(addprefix $(OBJ_DIR)/, $(LIB_SRCS:.cpp=.o))
ALL_DEPS  = $(shell find $(OBJ_DIR) | grep '.d$$')

ifeq "$(strip $(OBJ_DIR))" ""
  OBJ_DIR = .
endif

#中間ファイルを削除しない
.PRECIOUS: $(ALL_OBJS)

#======================================
default:
	@echo "MAIN_SRCS : $(MAIN_SRCS)"
	@make all --no-print-directory

all: $(TARGET) $(ALL_OBJS)

#実行ファイルを単一のmainとその他から作成
$(BIN_DIR)/%.out: $(LIB_OBJS) $(OBJ_DIR)/%.o
	@[ -d  $(dir $@) ] || mkdir -p $(dir $@)
	$(GCC) -o $@ $^ $(OPTION)

#OBJ_DIRにオブジェクトファイルを生成
$(OBJ_DIR)/%.o: %.cpp
	@[ -d  $(dir $@) ] || mkdir -p $(dir $@)
	$(GCC) $(CFLAGS) $(INCLUDES) -o $@ -c $< $(D_OPTION)

clean:
	@echo "rm -rf $(OBJ_DIR) $(BIN_DIR)"
	@rm -rf $(OBJ_DIR) $(BIN_DIR)

-include $(ALL_DEPS)
