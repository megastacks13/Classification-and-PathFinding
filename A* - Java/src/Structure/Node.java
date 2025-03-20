package Structure;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Node {

    List<Node> children;
    final int value;
    // List for controlling the depth of each parent node
    Map<Node, Integer> parentDepths = new HashMap<>();
    int row;
    int col;
    double heuristic;


    Node(int value, int row, int col, Node parent, List<Node> closedList) {
        this.value = value;
        this.row = row;
        this.col = col;
        addParent(parent, closedList);
        this.heuristic = Double.POSITIVE_INFINITY;
        this.children = new ArrayList<>();
    }

    public int getValue(){
        return value;
    }

    public void addParent(Node parent, List<Node> closedList){
        // If there is no parent, it is assumed to be the initial node, hence the depth is 0
        if (parent == null){
            parentDepths.put(null, 0);
        }
        // Any other way, is the minimum depth of the parent
        else
            parentDepths.put(parent, (int) parent.getMinimumDepth(closedList)[0]);
    }

    // We get the minimum depth from all the parent nodes that form part of the closed list
    public Object[] getMinimumDepth(List<Node> closedList){
        // This cannot happen
        if (closedList == null)
            throw new IllegalArgumentException("closedList is null");

        int minDepth = Integer.MAX_VALUE;
        Node minParent = null;
        for (Node node : closedList){
            if (parentDepths.containsKey(node)){
                if (minDepth > parentDepths.get(node)+1){
                    minDepth = parentDepths.get(node)+1;
                    minParent = node;
                }
            }
        }
        if (minDepth == Integer.MAX_VALUE && parentDepths.containsKey(null))
            minDepth = parentDepths.get(null);

        return new Object[]{minDepth, minParent};
    }

    public int getRow() {
        return row;
    }

    public int getCol() {
        return col;
    }

    public void addChild(Node child) {
        children.add(child);
    }

    public void modifyHeuristic(double heuristic) {
        this.heuristic = heuristic;
    }

    public double getHeuristic() {
        return heuristic;
    }

    public List<Node> getChildren() {
        return children;
    }

    // We override the default equals to compare cell values instead of pointers
    @Override
    public boolean equals(Object obj) {
        if (!(obj instanceof Node node)) return false;
        return node.getRow() == row && node.getCol() == col;
    }
}