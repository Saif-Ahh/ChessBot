import cv2
import threading
import time
import extract_pieces  # Assuming this module contains the get_pieces method
import pyautogui
import chess
import chess.engine
import keyboard

# constants
global ENGINE
ENGINE = chess.engine.SimpleEngine.popen_uci('Stockfish\\stockfish.exe')
pyautogui.MINIMUM_DURATION = 0
pyautogui.MINIMUM_SLEEP = 0
pyautogui.PAUSE = 0


def process_image(img_boards, img_gray, coord_list, col: str = None, converter: dict[str, str] = None) -> list[tuple]:
    # Get board cords
    if converter is None:
        if col == 'w':
            cord = extract_pieces.get_coords(img_boards, img_gray, ["board_finder\\bottom_right.png",
                                                                    "board_finder\\top_left.png"])
        else:
            cord = extract_pieces.get_coords(img_boards, img_gray, ["board_finder\\bottom_right_black.png",
                                                                    "board_finder\\top_left_black.png"])
    else:
        # Call the method to extract pieces
        cord = extract_pieces.get_pieces(img_boards, board_minx, board_miny)
    coord_list.extend(cord)


def get_fen(coords, board_minx, board_miny, col) -> str:
    # Turn coords into spots
    piece_dict = {}
    for i in range(1, len(coords), 2):
        x_pos = chr((coords[i][0] - board_minx) // 95 + 97)
        y_pos = 8 - ((coords[i][1] - board_miny) // 95)

        piece = coords[i - 1]
        piece_dict.setdefault(piece, [])
        piece_dict[piece].append(str(x_pos + str(y_pos)))

    fen_pieces = ''
    for rank in range(8, 0, -1):
        empty_squares = 0
        for file in 'abcdefgh':
            square = file + str(rank)
            piece = next((key for key, value in piece_dict.items() if square in value), '')
            if piece:
                if empty_squares > 0:
                    fen_pieces += str(empty_squares)
                    empty_squares = 0
                fen_pieces += piece
            else:
                empty_squares += 1
        if empty_squares > 0:
            fen_pieces += str(empty_squares)
        if rank > 1:
            fen_pieces += '/'
    return fen_pieces


def make_move(move_str: str, col: str, click: bool) -> None:
    # Invalid move
    if len(move_str) != 4 or len(move_str) != 5:
        pass

    # Get origin and dest square
    if col == 'w':
        org_x = ord(move_str[0])
        dest_x = ord(move_str[2])

        org_x = (org_x-96)*98+350
        dest_x = (dest_x-96)*98+350
        org_y = -95 * int(move_str[1]) + 1000
        dest_y = -95 * int(move_str[3]) + 1000
    else:
        org_x = (ord('h') - ord(move_str[0]) + 97)
        dest_x = (ord('h') - ord(move_str[2]) + 97)
        org_x = (org_x-96)*98+ 350
        dest_x = (dest_x-96)*98 + 350

        # Flip around move_str
        mov = 9 - int(move_str[1])
        dest_mov = 9 - int(move_str[3])
        org_y = -95 * mov + 1000
        dest_y = -95 * dest_mov + 1000

    # Right click to indicate position
    pyautogui.moveTo(org_x, org_y)
    pyautogui.mouseDown(button='left')
    pyautogui.mouseUp(button='left')
    time.sleep(0.1)
    if click:
        button = 'left'
    else:
        button = 'right'
    pyautogui.mouseDown(button=button)
    pyautogui.moveTo(dest_x, dest_y)
    pyautogui.mouseUp(button=button)


def get_best_move(board: chess.Board) -> str:
    global ENGINE
    try:
        info = ENGINE.analyse(board, chess.engine.Limit(time=0.01))
    except chess.engine.EngineTerminatedError as e:
        # Restart engine
        ENGINE = chess.engine.SimpleEngine.popen_uci('Stockfish\\stockfish.exe')
        return None
    return info['pv'][0]


def getScreen():
    game = pyautogui.screenshot()
    game = game.convert('RGBA')
    game.save('game.png')
    gamex = cv2.imread('game.png')
    return gamex


def cap_screen(col: str) -> str:
    board_coord = []
    coords = []
    img_board = getScreen()
    img_board_gray = cv2.cvtColor(img_board, cv2.COLOR_BGR2GRAY)

    # Create a thread
    b_converter = {'B': 'b_'}
    k_converter = {'K': 'k_'}
    n_converter = {'N': 'n_'}
    p_converter = {'P': 'p_'}
    r_converter = {'R': 'r_'}
    q_converter = {'Q': 'q_'}
    boord_thread = threading.Thread(target=process_image, args=(img_board, img_board_gray, board_coord, col, None))
    b_thread = threading.Thread(target=process_image, args=(img_board, img_board_gray, coords, '', b_converter))
    k_thread = threading.Thread(target=process_image, args=(img_board, img_board_gray, coords, '', k_converter))
    n_thread = threading.Thread(target=process_image, args=(img_board, img_board_gray, coords, '', n_converter))
    p_thread = threading.Thread(target=process_image, args=(img_board, img_board_gray, coords, '', p_converter))
    r_thread = threading.Thread(target=process_image, args=(img_board, img_board_gray, coords, '', r_converter))
    q_thread = threading.Thread(target=process_image, args=(img_board, img_board_gray, coords, '', q_converter))

    # Start the thread
    boord_thread.start()

    # get board coords
    boord_thread.join()
    board_minx = board_coord[0][0]
    board_miny = board_coord[0][1]

    # classify squares with ML model
    coords = extract_pieces.get_pieces(img_board, board_minx, board_miny)

    # Turn cords into fen
    fen = get_fen(coords, board_minx, board_miny, col) + ' ' + col + ' - - 0 1'
    if col == 'b':
        fen = fen.split()[0][::-1] + " b - - 0 1"
    return fen


def display_move(col: str, click: bool) -> None:
    t1 = time.time()
    fen = cap_screen(col)

    # Get best move and print out time taken
    perform_move(col, click, fen)
    print("Time taken:", time.time() - t1)


def perform_move(col: str, click: bool, fen: str) -> None:
    board = chess.Board(fen=fen)
    print(fen)
    move = get_best_move(board)
    if move is not None:
        make_move(str(move), col, click)


class MyThread:
    def __init__(self):
        self.is_running = False
        self.thread = None
        self.lock = threading.Lock()

    def start_thread(self):
        if not self.is_running:
            self.is_running = True
            self.thread = threading.Thread(target=self.loop_function)
            self.thread.start()
            time.sleep(0.1)

    def loop_function(self):
        old_fen = ''
        while self.is_running:
            
                print("Changed detected")
                with self.lock:
                    perform_move('w', True, new_fen)
                time.sleep(0.01)
                old_fen = cap_screen('w')

    def stop_thread(self):
        self.is_running = False
        self.thread.join()
        print("Thread has stopped.")


# default vals
my_thread_instance = MyThread()
while True:
    if keyboard.is_pressed('num 0'):
        display_move('w', False)
    elif keyboard.is_pressed('num 1'):
        display_move('b', False)
    elif keyboard.is_pressed('num 2'):
        display_move('w', True)
    elif keyboard.is_pressed('num 3'):
        display_move('b', True)
    elif keyboard.is_pressed('num 7'):
        my_thread_instance.start_thread()
    elif keyboard.is_pressed('num 9'):
        my_thread_instance.stop_thread()
    time.sleep(0.01)
