package com.example;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.List;
import org.junit.jupiter.api.Test;

class RefundSplitterTest {

    private final RefundSplitter splitter = new RefundSplitter();

    @Test
    void splits_proportionally_when_it_divides_evenly() {
        assertEquals(List.of(600L, 400L), splitter.split(1000L, List.of(1200L, 800L)));
    }

    @Test
    void hands_the_remainder_out_one_cent_at_a_time() {
        List<Long> shares = splitter.split(100L, List.of(100L, 100L, 100L));

        assertEquals(List.of(34L, 33L, 33L), shares);
        assertEquals(100L, shares.stream().mapToLong(Long::longValue).sum());
    }

    @Test
    void gives_every_item_zero_when_the_order_is_worth_nothing() {
        assertEquals(List.of(0L, 0L), splitter.split(0L, List.of(0L, 0L)));
    }

    @Test
    void refuses_a_refund_larger_than_the_items() {
        assertThrows(
                IllegalArgumentException.class, () -> splitter.split(1500L, List.of(1000L)));
    }

    @Test
    void refuses_negative_amounts() {
        assertThrows(IllegalArgumentException.class, () -> splitter.split(-1L, List.of(10L)));
        assertThrows(
                IllegalArgumentException.class, () -> splitter.split(10L, java.util.Arrays.asList(-5L)));
    }
}
