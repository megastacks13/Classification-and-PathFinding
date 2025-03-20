package Structure;

import java.util.*;

public class Table {

    private int[][] table;
    private int[] rootPositions;
    private final double diagonal;
    private List<int[]> route = new ArrayList<>();

    public Table(int number_rows, int number_cols) {
        if (number_rows < 1 || number_cols < 1) {
            throw new IllegalArgumentException("Invalid number of rows and columns");
        }
        table = new int[number_rows][number_cols];
        diagonal = Math.sqrt(table.length * table.length + table[0].length * table[0].length);
    }

    public void addProhibitedCell(int row, int col) {
        editCell(row, col, NodeManager.PROHIBITED_CELL);
    }

    public void addDangerousCell(int row, int col) {
        editCell(row, col, NodeManager.DANGEROUS_CELL);
    }

    public void removeCell(int row, int col) {
        editCell(row, col, NodeManager.NORMAL_CELL);
        // Removes from route in case of being a waypoint
        route.removeIf(rot -> rot[0] == row && rot[1] == col);
    }

    public void addStartCell(int row, int col) {
        editCell(row, col, NodeManager.INITIAL_CELL);
        // We store the cell that points to the initial value
        rootPositions = new int[]{row, col};
    }

    public void addFinalCell(int row, int col) {
        editCell(row, col, NodeManager.FINAL_CELL);
        // Last element of the route
        route.addLast(new int[]{row, col});
    }

    public void addWaypointCell(int row, int col) {
        editCell(row, col, NodeManager.WAYPOINT_CELL);
        // We add the waypoint to the route
        route.addLast(new int[]{row, col});
    }

    public boolean executeAlgorithm() {
        OrderedNodeList openList = new OrderedNodeList();
        List<Node> closedList = new ArrayList<>();
        Map<Node, Node> parentMap = new HashMap<>();

        int[] startNode = rootPositions;
        // Given the case that there are waypoints, we do the algorithm from each waypoint to the next
        for (int[] target : route) {
            if (!executeAStar(startNode, target, openList, closedList, parentMap)) {
                return false; // Not path found
            }
            // Update the values
            startNode = target;
            openList.clear();
            closedList.clear();
            parentMap.clear();
        }
        return true;
    }

    private boolean executeAStar(int[] startNode, int[] target, OrderedNodeList openList, List<Node> closedList, Map<Node, Node> parentMap) {
        // Create the initial node
        Node startNodeNode = NodeManager.createNewNode(null, startNode[0], startNode[1], table, closedList);
        // Mark it as so
        parentMap.put(startNodeNode, null);
        // Add it to the open list
        openList.insert(startNodeNode);

        while (!openList.isEmpty()) {
            // Remove element from open list
            Node currentNode = openList.poll();
            // Set it on the closed
            closedList.add(currentNode);
            // Calculate all the children of the node
            NodeManager.createChildren(currentNode, table, closedList);
            // Compute their heuristics and add them to the open list
            calculateChildrenHeuristics(currentNode, parentMap, openList, closedList);

            // Is it final?
            if (currentNode.getRow() == target[0] && currentNode.getCol() == target[1]) {
                markPath(currentNode, parentMap);
                return true; // Found path
            }

        }
        return false; // No path found
    }

    private void calculateChildrenHeuristics(Node parent, Map<Node, Node> parentMap, OrderedNodeList openList, List<Node> closedList) {
        // For all the children of a given parent
        for (Node child : parent.children) {
            // Get only the ones that are not in the closed list
            if (!closedList.contains(child)) {
                // Compute their heuristic
                Object[] obj = calculateHeuristic(child, closedList);
                double childHeuristic = (double) obj[0];
                Node minParent = (Node) obj[1];
                // Update the minimum heuristic
                child.modifyHeuristic(childHeuristic);
                // Insert the child in the open list
                openList.insert(child);
                // And the relation in the parent map
                parentMap.put(child, minParent);
            }
        }
    }

    private Object[] calculateHeuristic(Node node, List<Node> closedList) {
        Object[] obj = node.getMinimumDepth(closedList);

        double heuristic = (int) obj[0];
        Node parent = (Node) obj[1];
        // Add punishment if given the case
        if (node.value == NodeManager.DANGEROUS_CELL) {
            heuristic += 0.1 * diagonal;
        }
        // Return the optimal parent
        return new Object[]{heuristic, parent};
    }

    // We will go from the last node to the initial node using the parent map.
    // This map maps the children to the optimal parent, making it easy to route backwards
    private void markPath(Node finalNode, Map<Node, Node> parentMap) {
        Node currentNode = finalNode;
        while (currentNode != null) {
            int row = currentNode.getRow();
            int col = currentNode.getCol();
            // We don't want the Initial, Final or Waypoint cells to be overridden by the Walked cell
            if (table[row][col] != NodeManager.INITIAL_CELL && table[row][col] != NodeManager.FINAL_CELL &&
            table[row][col] != NodeManager.WAYPOINT_CELL) {
                // Different color for re-walked cells
                if (table[row][col] == NodeManager.WALKED_CELL)
                    table[row][col] = NodeManager.REWALKED_CELL;
                // Mark walked cell
                else
                    table[row][col] = NodeManager.WALKED_CELL;
            }
            // Go one step backwards
            currentNode = parentMap.get(currentNode);
        }
    }

    // Updates a cell
    private void editCell(int row, int col, int value) {
        if (row < 0 || row >= table.length || col < 0 || col >= table[0].length)
            throw new IllegalArgumentException("Invalid row or column");
        table[row][col] = value;
    }

    // Is a row and a column in bounds?
    public boolean isValidCell(int row, int col){
        return row >= 0 && row < table.length && col >= 0 && col < table[row].length;
    }

    // A getter I guess
    public int[][] getTable() {
        return table;
    }

    // Is a cell empty?
    public boolean isEmptyCell(int row, int col) {
        return table[row][col] == 0;
    }
}