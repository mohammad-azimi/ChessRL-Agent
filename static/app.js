let currentState = null;
let selectedSquare = null;
let busy = false;
let dragSession = null;
let boardOrientation = null;
let manualOrientation = false;

const files = ["a", "b", "c", "d", "e", "f", "g", "h"];

const pieceValues = {
  pawn: 1,
  knight: 3,
  bishop: 3,
  rook: 5,
  queen: 9,
  king: 0,
};

const startingPieceCounts = {
  pawn: 8,
  knight: 2,
  bishop: 2,
  rook: 2,
  queen: 1,
  king: 1,
};

const capturedPieceOrder = ["queen", "rook", "bishop", "knight", "pawn"];

const pieceUnicode = {
  white: {
    pawn: "♙",
    knight: "♘",
    bishop: "♗",
    rook: "♖",
    queen: "♕",
    king: "♔",
  },
  black: {
    pawn: "♟",
    knight: "♞",
    bishop: "♝",
    rook: "♜",
    queen: "♛",
    king: "♚",
  },
};

function setDisabledById(id, value) {
  const element = document.getElementById(id);

  if (element) {
    element.disabled = value;
  }
}

function setBusy(value, message = null) {
  busy = value;

  setDisabledById("newGameButton", value);
  setDisabledById("undoButton", value);
  setDisabledById("resetButton", value);
  setDisabledById("flipBoardButton", value);
  setDisabledById("copyFenButton", value);
  setDisabledById("copyPgnButton", value);

  const messageElement = document.getElementById("message");

  if (message) {
    messageElement.textContent = message;
  }

  if (value) {
    messageElement.classList.add("thinking");
  } else {
    messageElement.classList.remove("thinking");
  }
}

function squareName(fileIndex, rankIndex) {
  return files[fileIndex] + String(rankIndex + 1);
}

function getBoardOrientation(state) {
  if (boardOrientation) {
    return boardOrientation;
  }

  if (state && state.human_color) {
    return state.human_color;
  }

  return "white";
}

function setDefaultBoardOrientation(state) {
  if (!manualOrientation) {
    boardOrientation = state.human_color;
  }
}

function updateBoardOrientationText() {
  const element = document.getElementById("boardOrientationText");

  if (element) {
    element.textContent = boardOrientation || "-";
  }
}

function flipBoard() {
  if (!currentState) {
    return;
  }

  boardOrientation =
    getBoardOrientation(currentState) === "white" ? "black" : "white";
  manualOrientation = true;
  selectedSquare = null;

  renderBoard(currentState);
  updateBoardOrientationText();

  document.getElementById("message").textContent = "Board flipped.";
}

function getDisplaySquares(orientation) {
  const fileIndexes =
    orientation === "white"
      ? [0, 1, 2, 3, 4, 5, 6, 7]
      : [7, 6, 5, 4, 3, 2, 1, 0];

  const rankIndexes =
    orientation === "white"
      ? [7, 6, 5, 4, 3, 2, 1, 0]
      : [0, 1, 2, 3, 4, 5, 6, 7];

  const squares = [];

  for (const rankIndex of rankIndexes) {
    for (const fileIndex of fileIndexes) {
      squares.push(squareName(fileIndex, rankIndex));
    }
  }

  return squares;
}

function getLastMoveSquares(state) {
  if (!state || !state.move_history || state.move_history.length === 0) {
    return [];
  }

  const lastMove = state.move_history[state.move_history.length - 1].uci;

  return [lastMove.slice(0, 2), lastMove.slice(2, 4)];
}

function shouldShowFileLabel(square, orientation) {
  const rank = square[1];

  if (orientation === "white") {
    return rank === "1";
  }

  return rank === "8";
}

function shouldShowRankLabel(square, orientation) {
  const file = square[0];

  if (orientation === "white") {
    return file === "a";
  }

  return file === "h";
}

function isHumanTurn() {
  return currentState && currentState.turn === currentState.human_color;
}

function getAgentColor(state) {
  return state.human_color === "white" ? "black" : "white";
}

function getBottomPlayerColor(state) {
  return getBoardOrientation(state);
}

function getTopPlayerColor(state) {
  return getBoardOrientation(state) === "white" ? "black" : "white";
}

function isLegalDestination(fromSquare, toSquare) {
  if (!currentState || !fromSquare) {
    return false;
  }

  return currentState.legal_moves.some((move) => {
    return move.startsWith(fromSquare + toSquare);
  });
}

function getLegalMoveForSquares(fromSquare, toSquare) {
  if (!currentState) {
    return null;
  }

  const exactMove = currentState.legal_moves.find(
    (move) => move === fromSquare + toSquare,
  );

  if (exactMove) {
    return exactMove;
  }

  const promotionMove = currentState.legal_moves.find((move) => {
    return move.startsWith(fromSquare + toSquare);
  });

  return promotionMove || null;
}

function updateDifficultyVisuals(difficulty) {
  document.querySelectorAll(".difficulty-item").forEach((item) => {
    item.classList.toggle("active", item.dataset.difficulty === difficulty);
  });
}

function countPiecesByColorAndType(state) {
  const counts = {
    white: {
      pawn: 0,
      knight: 0,
      bishop: 0,
      rook: 0,
      queen: 0,
      king: 0,
    },
    black: {
      pawn: 0,
      knight: 0,
      bishop: 0,
      rook: 0,
      queen: 0,
      king: 0,
    },
  };

  for (const piece of Object.values(state.pieces)) {
    counts[piece.color][piece.type] += 1;
  }

  return counts;
}

function calculateMaterialScore(state) {
  const counts = countPiecesByColorAndType(state);

  let whiteScore = 0;
  let blackScore = 0;

  for (const [pieceType, value] of Object.entries(pieceValues)) {
    whiteScore += counts.white[pieceType] * value;
    blackScore += counts.black[pieceType] * value;
  }

  return {
    white: whiteScore,
    black: blackScore,
    balance: whiteScore - blackScore,
  };
}

function getCapturedPiecesForPlayer(state, playerColor) {
  const counts = countPiecesByColorAndType(state);
  const opponentColor = playerColor === "white" ? "black" : "white";
  const capturedPieces = [];

  for (const pieceType of capturedPieceOrder) {
    const missingCount =
      startingPieceCounts[pieceType] - counts[opponentColor][pieceType];

    for (let i = 0; i < missingCount; i += 1) {
      capturedPieces.push({
        type: pieceType,
        color: opponentColor,
        unicode: pieceUnicode[opponentColor][pieceType],
      });
    }
  }

  return capturedPieces;
}

function getMaterialAdvantageForPlayer(state, playerColor) {
  const material = calculateMaterialScore(state);

  if (playerColor === "white" && material.balance > 0) {
    return "+" + material.balance;
  }

  if (playerColor === "black" && material.balance < 0) {
    return "+" + Math.abs(material.balance);
  }

  return "";
}

function renderCapturedPieces(elementId, capturedPieces) {
  const element = document.getElementById(elementId);
  element.innerHTML = "";

  for (const piece of capturedPieces) {
    const pieceElement = document.createElement("span");
    pieceElement.className = "captured-piece";
    pieceElement.textContent = piece.unicode;
    element.appendChild(pieceElement);
  }
}

function renderPlayerCards(state) {
  const topColor = getTopPlayerColor(state);
  const bottomColor = getBottomPlayerColor(state);
  const agentColor = getAgentColor(state);

  const topCard = document.getElementById("topPlayerCard");
  const bottomCard = document.getElementById("bottomPlayerCard");

  topCard.classList.toggle("active", state.turn === topColor);
  bottomCard.classList.toggle("active", state.turn === bottomColor);

  document.getElementById("topPlayerAvatar").textContent =
    topColor === "white" ? "♙" : "♟";
  document.getElementById("bottomPlayerAvatar").textContent =
    bottomColor === "white" ? "♙" : "♟";

  if (topColor === state.human_color) {
    document.getElementById("topPlayerName").textContent = "You";
    document.getElementById("topPlayerMeta").textContent =
      `${topColor} • human`;
  } else {
    document.getElementById("topPlayerName").textContent = state.agent_label;
    document.getElementById("topPlayerMeta").textContent =
      `${agentColor} • ${state.difficulty_label}`;
  }

  if (bottomColor === state.human_color) {
    document.getElementById("bottomPlayerName").textContent = "You";
    document.getElementById("bottomPlayerMeta").textContent =
      `${bottomColor} • human`;
  } else {
    document.getElementById("bottomPlayerName").textContent = state.agent_label;
    document.getElementById("bottomPlayerMeta").textContent =
      `${agentColor} • ${state.difficulty_label}`;
  }

  renderCapturedPieces(
    "topCapturedPieces",
    getCapturedPiecesForPlayer(state, topColor),
  );
  renderCapturedPieces(
    "bottomCapturedPieces",
    getCapturedPiecesForPlayer(state, bottomColor),
  );

  document.getElementById("topMaterialAdvantage").textContent =
    getMaterialAdvantageForPlayer(state, topColor);

  document.getElementById("bottomMaterialAdvantage").textContent =
    getMaterialAdvantageForPlayer(state, bottomColor);
}

function createDragGhost(piece, clientX, clientY) {
  const boardElement = document.getElementById("board");
  const boardRect = boardElement.getBoundingClientRect();
  const squareSize = boardRect.width / 8;

  const ghost = document.createElement("div");
  ghost.className = "piece";
  ghost.classList.add(piece.color);
  ghost.textContent = piece.unicode;

  ghost.style.position = "fixed";
  ghost.style.left = clientX + "px";
  ghost.style.top = clientY + "px";
  ghost.style.zIndex = "9999";
  ghost.style.pointerEvents = "none";
  ghost.style.fontSize = Math.round(squareSize * 0.82) + "px";
  ghost.style.transform = "translate(-50%, -50%)";
  ghost.style.opacity = "0.96";

  document.body.appendChild(ghost);

  return ghost;
}

function moveDragGhost(clientX, clientY) {
  if (!dragSession || !dragSession.ghost) {
    return;
  }

  dragSession.ghost.style.left = clientX + "px";
  dragSession.ghost.style.top = clientY + "px";
}

function removeDragGhost() {
  if (dragSession && dragSession.ghost) {
    dragSession.ghost.remove();
    dragSession.ghost = null;
  }
}

function getSquareFromPoint(clientX, clientY) {
  const element = document.elementFromPoint(clientX, clientY);

  if (!element) {
    return null;
  }

  const squareElement = element.closest(".square");

  if (!squareElement) {
    return null;
  }

  return squareElement.dataset.square || null;
}

function renderBoard(state) {
  const boardElement = document.getElementById("board");
  boardElement.innerHTML = "";

  const orientation = getBoardOrientation(state);
  const displaySquares = getDisplaySquares(orientation);
  const pieceMap = state.pieces;
  const lastMoveSquares = getLastMoveSquares(state);

  for (const square of displaySquares) {
    const fileIndex = files.indexOf(square[0]);
    const rankIndex = Number(square[1]) - 1;
    const piece = pieceMap[square];

    const squareElement = document.createElement("div");
    squareElement.className = "square";
    squareElement.dataset.square = square;

    const isLight = (fileIndex + rankIndex) % 2 === 0;
    squareElement.classList.add(isLight ? "light" : "dark");

    if (lastMoveSquares.includes(square)) {
      squareElement.classList.add("last-move");
    }

    if (selectedSquare === square) {
      squareElement.classList.add("selected");
    }

    if (
      dragSession &&
      dragSession.isDragging &&
      dragSession.fromSquare === square
    ) {
      squareElement.classList.add("dragging");
    }

    if (selectedSquare && isLegalDestination(selectedSquare, square)) {
      if (piece) {
        squareElement.classList.add("capture");
      } else {
        squareElement.classList.add("legal");
      }
    }

    squareElement.addEventListener("pointerdown", (event) => {
      handlePointerDown(event, square);
    });

    if (shouldShowRankLabel(square, orientation)) {
      const rankLabel = document.createElement("div");
      rankLabel.className = "coord-rank";
      rankLabel.textContent = square[1];
      squareElement.appendChild(rankLabel);
    }

    if (shouldShowFileLabel(square, orientation)) {
      const fileLabel = document.createElement("div");
      fileLabel.className = "coord-file";
      fileLabel.textContent = square[0];
      squareElement.appendChild(fileLabel);
    }

    if (piece) {
      const pieceElement = document.createElement("div");
      pieceElement.className = "piece";
      pieceElement.classList.add(piece.color);
      pieceElement.textContent = piece.unicode;
      squareElement.appendChild(pieceElement);
    }

    boardElement.appendChild(squareElement);
  }
}

function renderMoveHistory(state) {
  const historyElement = document.getElementById("moveHistory");
  historyElement.innerHTML = "";

  if (!state.move_history || state.move_history.length === 0) {
    const empty = document.createElement("div");
    empty.className = "move-pill";
    empty.textContent = "No moves yet.";
    historyElement.appendChild(empty);
    return;
  }

  for (let i = 0; i < state.move_history.length; i += 2) {
    const whiteMove = state.move_history[i];
    const blackMove = state.move_history[i + 1];

    const row = document.createElement("div");
    row.className = "history-row";

    const moveNumber = document.createElement("strong");
    moveNumber.textContent = String(Math.floor(i / 2) + 1) + ".";

    const whiteCell = document.createElement("div");
    whiteCell.textContent = whiteMove
      ? whiteMove.san + " (" + whiteMove.side + ")"
      : "-";

    const blackCell = document.createElement("div");
    blackCell.textContent = blackMove
      ? blackMove.san + " (" + blackMove.side + ")"
      : "-";

    row.appendChild(moveNumber);
    row.appendChild(whiteCell);
    row.appendChild(blackCell);

    historyElement.appendChild(row);
  }

  historyElement.scrollTop = historyElement.scrollHeight;
}

function renderStatus(state) {
  document.getElementById("agentBadge").textContent =
    state.difficulty_label + " • " + state.agent_label;
  document.getElementById("difficultyText").textContent =
    state.difficulty_label;
  document.getElementById("engineText").textContent = state.agent_label;
  document.getElementById("turnText").textContent = state.turn;
  document.getElementById("humanColorText").textContent = state.human_color;
  document.getElementById("boardOrientationText").textContent =
    getBoardOrientation(state);
  document.getElementById("resultText").textContent = state.result || "-";
  document.getElementById("reasonText").textContent = state.reason || "-";

  const legalMovesElement = document.getElementById("legalMoves");
  legalMovesElement.innerHTML = "";

  for (const move of state.legal_moves) {
    const moveElement = document.createElement("div");
    moveElement.className = "move-pill";
    moveElement.textContent = move;
    legalMovesElement.appendChild(moveElement);
  }

  document.getElementById("difficultySelect").value = state.difficulty_key;
  updateDifficultyVisuals(state.difficulty_key);
  renderMoveHistory(state);
  renderPlayerCards(state);

  if (state.game_over) {
    document.getElementById("message").innerHTML =
      "<span class='game-over'>Game over.</span> Result: " + state.result;
  }
}

function renderState(state) {
  currentState = state;
  setDefaultBoardOrientation(state);
  renderBoard(state);
  renderStatus(state);
}

async function loadState() {
  selectedSquare = null;

  const response = await fetch("/api/state", {
    cache: "no-store",
  });

  const data = await response.json();
  renderState(data);
}

async function startNewGame() {
  selectedSquare = null;
  manualOrientation = false;
  setBusy(true, "Starting a new game...");

  const difficulty = document.getElementById("difficultySelect").value;
  const color = document.getElementById("colorSelect").value;

  try {
    const response = await fetch("/api/new-game", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        difficulty: difficulty,
        human_color: color,
      }),
    });

    const data = await response.json();

    if (!response.ok) {
      document.getElementById("message").textContent =
        data.error || "Could not start game.";
      return;
    }

    renderState(data.state);
    document.getElementById("message").textContent = data.message;
  } finally {
    setBusy(false);
  }
}

async function resetBoard() {
  selectedSquare = null;
  manualOrientation = false;
  setBusy(true, "Resetting board...");

  const difficulty = document.getElementById("difficultySelect").value;
  const color = document.getElementById("colorSelect").value;

  try {
    const response = await fetch("/api/new-game", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        difficulty: difficulty,
        human_color: color,
      }),
    });

    const data = await response.json();

    if (!response.ok) {
      document.getElementById("message").textContent =
        data.error || "Could not reset board.";
      return;
    }

    renderState(data.state);
    document.getElementById("message").textContent =
      "Board reset. " + data.message;
  } finally {
    setBusy(false);
  }
}

async function undoMove() {
  selectedSquare = null;
  setBusy(true, "Undoing last move...");

  try {
    const response = await fetch("/api/undo", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    });

    const data = await response.json();

    if (!response.ok) {
      if (data.state) {
        renderState(data.state);
      }

      document.getElementById("message").textContent =
        data.error || "Could not undo move.";
      return;
    }

    renderState(data.state);
    document.getElementById("message").textContent = data.message;
  } finally {
    setBusy(false);
  }
}

async function makeMove(move) {
  setBusy(true, "Thinking...");

  try {
    const response = await fetch("/api/move", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ move }),
    });

    const data = await response.json();

    if (!response.ok) {
      selectedSquare = null;

      if (data.state) {
        renderState(data.state);
      } else {
        await loadState();
      }

      document.getElementById("message").textContent =
        data.error || "Illegal move.";
      return;
    }

    selectedSquare = null;
    renderState(data.state);

    let message = "You played " + data.human_move + ".";

    if (data.agent_move) {
      message += " Agent played " + data.agent_move + ".";
    }

    document.getElementById("message").textContent = message;
  } finally {
    setBusy(false);
  }
}

function attemptMove(fromSquare, toSquare) {
  if (busy || !currentState || currentState.game_over) {
    return;
  }

  if (!isHumanTurn()) {
    document.getElementById("message").textContent = "Wait for the agent move.";
    return;
  }

  const legalMove = getLegalMoveForSquares(fromSquare, toSquare);

  if (!legalMove) {
    document.getElementById("message").textContent = "Illegal move.";
    selectedSquare = null;
    renderBoard(currentState);
    return;
  }

  makeMove(legalMove);
}

function handleSquareTap(square) {
  if (busy) {
    return;
  }

  if (!currentState || currentState.game_over) {
    return;
  }

  if (!isHumanTurn()) {
    document.getElementById("message").textContent = "Wait for the agent move.";
    return;
  }

  const piece = currentState.pieces[square];

  if (!selectedSquare) {
    if (!piece || piece.color !== currentState.human_color) {
      document.getElementById("message").textContent =
        "Select one of your pieces.";
      return;
    }

    selectedSquare = square;
    renderBoard(currentState);
    return;
  }

  if (selectedSquare === square) {
    selectedSquare = null;
    renderBoard(currentState);
    return;
  }

  if (piece && piece.color === currentState.human_color) {
    selectedSquare = square;
    renderBoard(currentState);
    return;
  }

  attemptMove(selectedSquare, square);
}

function handlePointerDown(event, square) {
  if (busy || !currentState || currentState.game_over) {
    return;
  }

  if (event.button !== undefined && event.button !== 0) {
    return;
  }

  event.preventDefault();

  const piece = currentState.pieces[square];
  const canDragPiece =
    piece && piece.color === currentState.human_color && isHumanTurn();

  dragSession = {
    startX: event.clientX,
    startY: event.clientY,
    currentX: event.clientX,
    currentY: event.clientY,
    fromSquare: square,
    piece: piece,
    canDragPiece: Boolean(canDragPiece),
    isDragging: false,
    ghost: null,
  };

  document.addEventListener("pointermove", handlePointerMove, {
    passive: false,
  });
  document.addEventListener("pointerup", handlePointerUp, { passive: false });
  document.addEventListener("pointercancel", handlePointerCancel, {
    passive: false,
  });
}

function handlePointerMove(event) {
  if (!dragSession) {
    return;
  }

  event.preventDefault();

  dragSession.currentX = event.clientX;
  dragSession.currentY = event.clientY;

  const deltaX = event.clientX - dragSession.startX;
  const deltaY = event.clientY - dragSession.startY;
  const distance = Math.sqrt(deltaX * deltaX + deltaY * deltaY);

  if (dragSession.canDragPiece && !dragSession.isDragging && distance > 7) {
    dragSession.isDragging = true;
    selectedSquare = dragSession.fromSquare;
    renderBoard(currentState);
    dragSession.ghost = createDragGhost(
      dragSession.piece,
      event.clientX,
      event.clientY,
    );
  }

  if (dragSession.isDragging) {
    moveDragGhost(event.clientX, event.clientY);
  }
}

function handlePointerUp(event) {
  if (!dragSession) {
    return;
  }

  event.preventDefault();

  const session = dragSession;
  const wasDragging = session.isDragging;

  cleanupPointerDrag();

  if (wasDragging) {
    const targetSquare = getSquareFromPoint(event.clientX, event.clientY);

    selectedSquare = null;

    if (!targetSquare) {
      renderBoard(currentState);
      return;
    }

    if (targetSquare === session.fromSquare) {
      renderBoard(currentState);
      return;
    }

    attemptMove(session.fromSquare, targetSquare);
    return;
  }

  handleSquareTap(session.fromSquare);
}

function handlePointerCancel() {
  cleanupPointerDrag();
  selectedSquare = null;

  if (currentState) {
    renderBoard(currentState);
  }
}

function cleanupPointerDrag() {
  removeDragGhost();

  document.removeEventListener("pointermove", handlePointerMove);
  document.removeEventListener("pointerup", handlePointerUp);
  document.removeEventListener("pointercancel", handlePointerCancel);

  dragSession = null;
}

async function copyTextToClipboard(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    const textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.style.position = "fixed";
    textarea.style.opacity = "0";
    document.body.appendChild(textarea);
    textarea.select();

    const copied = document.execCommand("copy");
    textarea.remove();

    return copied;
  }
}

async function copyFen() {
  if (!currentState) {
    return;
  }

  const copied = await copyTextToClipboard(currentState.fen);

  document.getElementById("message").textContent = copied
    ? "FEN copied to clipboard."
    : "Could not copy FEN.";
}

function buildPgnFromHistory(state) {
  const lines = [
    '[Event "ChessRL Agent Game"]',
    '[Site "Local Flask App"]',
    `[Difficulty "${state.difficulty_label}"]`,
    `[Engine "${state.agent_label}"]`,
    `[HumanColor "${state.human_color}"]`,
    `[Result "${state.result || "*"}"]`,
    "",
  ];

  const moveTokens = [];

  for (let i = 0; i < state.move_history.length; i += 2) {
    const moveNumber = Math.floor(i / 2) + 1;
    const whiteMove = state.move_history[i];
    const blackMove = state.move_history[i + 1];

    let token = `${moveNumber}.`;

    if (whiteMove) {
      token += ` ${whiteMove.san}`;
    }

    if (blackMove) {
      token += ` ${blackMove.san}`;
    }

    moveTokens.push(token);
  }

  const result = state.result || "*";
  lines.push(
    moveTokens.join(" ") + (moveTokens.length > 0 ? " " : "") + result,
  );

  return lines.join("\n");
}

async function copyPgn() {
  if (!currentState) {
    return;
  }

  const pgn = buildPgnFromHistory(currentState);
  const copied = await copyTextToClipboard(pgn);

  document.getElementById("message").textContent = copied
    ? "PGN copied to clipboard."
    : "Could not copy PGN.";
}

document
  .getElementById("newGameButton")
  .addEventListener("click", startNewGame);
document.getElementById("undoButton").addEventListener("click", undoMove);
document.getElementById("resetButton").addEventListener("click", resetBoard);
document.getElementById("flipBoardButton").addEventListener("click", flipBoard);
document.getElementById("copyFenButton").addEventListener("click", copyFen);
document.getElementById("copyPgnButton").addEventListener("click", copyPgn);

document
  .getElementById("difficultySelect")
  .addEventListener("change", (event) => {
    updateDifficultyVisuals(event.target.value);
  });

document.querySelectorAll(".difficulty-item").forEach((item) => {
  item.addEventListener("click", () => {
    document.getElementById("difficultySelect").value = item.dataset.difficulty;
    updateDifficultyVisuals(item.dataset.difficulty);
  });
});

loadState();
