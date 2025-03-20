package Structure;

import java.util.*;

public class NodeManager {

    public final static int REWALKED_CELL = -2;
    public final static int WALKED_CELL = -1;
    public final static int NORMAL_CELL = 0;
    public final static int INITIAL_CELL = 1;
    public final static int FINAL_CELL = 2;
    public final static int DANGEROUS_CELL = 3;
    public final static int PROHIBITED_CELL = 4;
    public final static int WAYPOINT_CELL = 5;

    static List<Node> nodes = new ArrayList<>();

    public static void createChildren(Node parent, int[][] table, List<Node> closedList) {
        int[][] directions = new int[][]{
                {0, 1},  // Right
                {0, -1}, // Left
                {-1, 0}, // Up
                {1, 0},  // Down
                {-1, 1}, // Up-right
                {1, 1},  // Down-right
                {-1, -1}, // Up-left
                {1, -1} // Down-left
        };

        for (int[] dir : directions) {
            int newRow = parent.getRow() + dir[0];
            int newCol = parent.getCol() + dir[1];

            if (isValid(newRow, newCol, table) && table[newRow][newCol] != PROHIBITED_CELL)
                createNewNode(parent, newRow, newCol, table, closedList);

        }
    }

    public static Node createNewNode(Node parent, int newRow, int newCol, int[][] table, List<Node> closedList) {
        Node childNode;
        // Initial Node
        if (parent == null){
            childNode = new Node(table[newRow][newCol], newRow, newCol, null, null);
            nodes.add(childNode);
            return childNode;
        }

        // Check if the node already exists
        for (Node node : nodes) {
            if (node.getRow() == newRow && node.getCol() == newCol) {
                // Update the information of said node
                parent.addChild(node);
                node.addParent(parent, closedList);
                return node;
            }
        }

        // Create new node if not
        childNode = new Node(table[newRow][newCol], newRow, newCol, parent, closedList);
        parent.addChild(childNode);
        nodes.add(childNode);
        return childNode;
    }


    // Helper method to check if a cell is valid
    private static boolean isValid(int row, int col, int[][] table) {
        return row >= 0 && row < table.length && col >= 0 && col < table[0].length;
    }

    // NodePosition class to hold the node and its depth
    private static class NodePosition {
        Node node;
        int depth;

        NodePosition(Node node, int depth) {
            this.node = node;
            this.depth = depth;
        }
    }


}