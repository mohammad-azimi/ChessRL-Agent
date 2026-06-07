let currentState = null;
let selectedSquare = null;
let busy = false;

const files = ["a", "b", "c", "d", "e", "f", "g", "h"];

function setBusy(value, message = null) {
  busy = value;

  document.getElementById("newGameButton").disabled = value;
  document.getElementById("undoButton").disabled = value;
  document.getElementById("resetButton").disabled = value;

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

function renderBoard(state) {
  const boardElement = document.getElementById("board");
  boardElement.innerHTML = "";

  const displaySquares = getDisplaySquares(state.human_color);
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

    if (selectedSquare && isLegalDestination(selectedSquare, square)) {
      if (piece) {
        squareElement.classList.add("capture");
      } else {
        squareElement.classList.add("legal");
      }
    }

    const canDragPiece =
      piece &&
      piece.color === state.human_color &&
      isHumanTurn() &&
      !state.game_over &&
      !busy;

    squareElement.draggable = Boolean(canDragPiece);

    squareElement.addEventListener("click", () => handleSquareClick(square));

    squareElement.addEventListener("dragstart", (event) => {
      if (!canDragPiece) {
        event.preventDefault();
        return;
      }

      selectedSquare = square;
      event.dataTransfer.setData("text/plain", square);
      squareElement.classList.add("dragging");
      renderBoard(currentState);
    });

    squareElement.addEventListener("dragend", () => {
      squareElement.classList.remove("dragging");
    });

    squareElement.addEventListener("dragover", (event) => {
      event.preventDefault();
    });

    squareElement.addEventListener("drop", (event) => {
      event.preventDefault();

      const fromSquare = event.dataTransfer.getData("text/plain");
      const toSquare = square;

      if (!fromSquare) {
        return;
      }

      attemptMove(fromSquare, toSquare);
    });

    if (shouldShowRankLabel(square, state.human_color)) {
      const rankLabel = document.createElement("div");
      rankLabel.className = "coord-rank";
      rankLabel.textContent = square[1];
      squareElement.appendChild(rankLabel);
    }

    if (shouldShowFileLabel(square, state.human_color)) {
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

  if (state.game_over) {
    document.getElementById("message").innerHTML =
      "<span class='game-over'>Game over.</span> Result: " + state.result;
  }
}

function renderState(state) {
  currentState = state;
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

function handleSquareClick(square) {
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

document
  .getElementById("newGameButton")
  .addEventListener("click", startNewGame);
document.getElementById("undoButton").addEventListener("click", undoMove);
document.getElementById("resetButton").addEventListener("click", resetBoard);

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
